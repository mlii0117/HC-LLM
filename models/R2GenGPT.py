import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, \
    BertModel
from pycocoevalcap.tri_consistency_constraints import RKDLoss, MSE, TripletLoss, SD_Constration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.metrics_clinical import CheXbertMetrics
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import copy
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

#(Optional)
class QFormer(nn.Module):
    def __init__(self, hidden_dim, query_dim, bert_model_name='./bert-base-uncased/',
                 is_lock=True):
        super(QFormer, self).__init__()
        self.image_projection = nn.Linear(hidden_dim, 768)
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        if is_lock:
            for name, param in self.encoder.named_parameters():
                param.requires_grad_(False)

    def forward(self, img_features, queries):
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=img_features.size(0))
        projected_img_features = self.image_projection(img_features)
        inputs = torch.cat([cls_tokens, projected_img_features, queries], dim=1)
        attention_mask = torch.ones(inputs.size()[:2], device=inputs.device)
        outputs = self.encoder(inputs_embeds=inputs, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, -32:, :]

#(Optional)
class VisualEncoderWithQFormer(nn.Module):
    def __init__(self, image_encoder, qformer):
        super(VisualEncoderWithQFormer, self).__init__()
        self.image_encoder = image_encoder
        self.qformer = qformer

    def forward(self, img_tensor):
        base_features = self.image_encoder(img_tensor)['last_hidden_state']
        queries = torch.randn(img_tensor.shape[0], 32, 768).to(img_tensor.device)  # adjust query dim
        output_features = self.qformer(base_features, queries)
        return output_features

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        self.cross_att = SD_Constration(hidden_dim=4096)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chexbert_metrics = CheXbertMetrics('./chexbert.pth', args.batch_size, torch.device(args.device))
        self.tripletloss = TripletLoss(self._device)
        self.RKDLoss = RKDLoss(25, 50)
        self.loss_sim = MSE()
        # Add Q-former (Optional)
        # qformer = QFormer(hidden_dim=self.visual_encoder.config.hidden_size, query_dim=768)  # adjust query dim
        # self.visual_encoder_with_qformer = VisualEncoderWithQFormer(self.visual_encoder, qformer)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()

            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)  # 分词器
        self.llama_tokenizer.pad_token_id = 0

        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            print("Begining to load LLAMA")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )

        print("Successfully to load LLAMA")
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            print("Beginning to load Input_Embedding")

            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        image_embeds = []
        image_embeds_pooler = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed_pooler = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
                image_embed_pooler = self.visual_encoder(image)['pooler_output'].to(device)
            image_embeds.append(image_embed)
            image_embeds_pooler.append(image_embed_pooler)
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)

        image_embeds_pool = torch.stack(image_embeds_pooler).mean(0)
        inputs_llama_pooler = self.llama_proj(image_embeds_pool)

        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, inputs_llama_pooler, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        batch_size = img_embeds.shape[0]
        p_before = 'Human: <Img>'
        p_after = '</Img> Generate a comprehensive and detailed diagnosis report for this chest xray image. \nAssistant:'

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def prompt_wrap_context(self, img_embeds, atts_img, context_input_text):
        batch_size = img_embeds.shape[0]

        p_before = 'Human: <Img>'
        p_mid = '</Img> Generate a comprehensive and detailed diagnosis report for this chest xray image.\nHere is the historical diagnosis report for reference: <Report>'
        p_after = '</Report> \nAssistant:'

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_mid_tokens = self.llama_tokenizer(
            p_mid, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        context_tokens = self.llama_tokenizer(
            context_input_text, return_tensors="pt", add_special_tokens=False, padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length).to(img_embeds.device)

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_mid_embeds = self.embed_tokens(p_mid_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        context_embeds = self.embed_tokens(context_tokens.input_ids)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_mid_embeds, context_embeds, p_after_embeds],
                                       dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def prompt_wrap_all(self, img_embeds, atts_img, context_img_embeds, context_input_text):
        batch_size = img_embeds.shape[0]
        p_before = 'Human: <Img>'
        p_mid = '</Img> Generate a comprehensive and detailed diagnosis report for this chest xray image.\nHere is the historical chest xray image <Img>'
        p_mid1 = '</Img> and diagnosis report: <report>'
        p_after = '</report> \nAssistant:'

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_mid_tokens = self.llama_tokenizer(
            p_mid, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_mid_tokens1 = self.llama_tokenizer(
            p_mid1, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        context_tokens = self.llama_tokenizer(
            context_input_text, return_tensors="pt", add_special_tokens=False, padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length).to(img_embeds.device)

        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_mid_embeds = self.embed_tokens(p_mid_tokens.input_ids).expand(batch_size, -1, -1)
        p_mid_embeds1 = self.embed_tokens(p_mid_tokens1.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        context_embeds = self.embed_tokens(context_tokens.input_ids)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_mid_embeds, context_img_embeds, p_mid_embeds1, context_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, img_embeds_pooler,atts_img = self.encode_img(image)
        current_img_embeds = self.layer_norm(img_embeds)
        img_embeds_pooler = self.layer_norm(img_embeds_pooler)
        context_input_text = samples["context_input_text"]

        context_image = samples["context_image"]
        context_img_embeds, context_embeds_pooler,context_atts_img = self.encode_img(context_image)
        context_img_embeds = self.layer_norm(context_img_embeds)
        context_embeds_pooler = self.layer_norm(context_embeds_pooler)

        img_embeds_llama, atts_img = self.prompt_wrap_context(current_img_embeds, atts_img, context_input_text)

        text = [t + self.end_sym for t in samples["input_text"]]
        context_text = [t + self.end_sym for t in context_input_text]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        to_regress_context_tokens = self.llama_tokenizer(
            context_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        context_targets = to_regress_context_tokens.input_ids.masked_fill(
            to_regress_context_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds_context = self.embed_tokens(to_regress_context_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds_llama, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        inputs_embeds_context = torch.cat([bos_embeds, img_embeds_llama, to_regress_embeds_context], dim=1)
        attention_mask_context = torch.cat([atts_bos, atts_img, to_regress_context_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        outputs_context = self.llama_model(
            inputs_embeds=inputs_embeds_context,
            attention_mask=attention_mask_context,
            output_hidden_states=True,
            return_dict=True,
        )

        current_text_embed = self.layer_norm(torch.max(outputs.hidden_states[-1][:,-self.hparams.max_length:,:],dim=1)[0])
        prior_text_embed = self.layer_norm(torch.max(outputs_context.hidden_states[-1][:,-self.hparams.max_length:,:],dim=1)[0])

        img_shared_c, img_shared_p, img_private_c, img_private_p = self.cross_att(img_embeds_pooler, context_embeds_pooler)
        txt_shared_c, txt_shared_p, txt_private_c, txt_private_p = self.cross_att(current_text_embed, prior_text_embed)

        image_similarity_loss = self.loss_sim(img_shared_c, img_shared_p)
        text_similarity_loss = self.loss_sim(txt_shared_c, txt_shared_p)

        rkdloss =  self.RKDLoss(torch.cat([(img_shared_c+img_shared_p).unsqueeze(1), img_private_c.unsqueeze(1), img_private_p.unsqueeze(1)],dim=1),
                                torch.cat([(txt_shared_c+txt_shared_p).unsqueeze(1), txt_private_c.unsqueeze(1), txt_private_p.unsqueeze(1)],dim=1))

        tripletloss = self.tripletloss(torch.cat([img_shared_c+img_shared_p, img_private_c, img_private_p],dim=0),
                                       torch.cat([txt_shared_c+txt_shared_p, txt_private_c, txt_private_p],dim=0))

        loss_rkd = 1.0 * rkdloss + 0.8*image_similarity_loss + 0.8*text_similarity_loss + 1.0 * tripletloss
        loss = outputs.loss + loss_rkd
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['ROUGE_L']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        current_img_embeds, _, atts_img = self.encode_img(image)
        current_img_embeds = self.layer_norm(current_img_embeds)

        context_image = samples["context_image"]
        context_img_embeds, _, context_atts_img = self.encode_img(context_image)
        context_img_embeds = self.layer_norm(context_img_embeds)
        context_report = samples["context_input_text"]

        current_img_embeds, atts_img = self.prompt_wrap_context(current_img_embeds, atts_img, context_report)

        batch_size = current_img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, current_img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        generated_tokens = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature
        )

        hypo = [self.decode(i) for i in generated_tokens]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace('<s>', '')  # 移除起始标记

        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref1 = {k: [v] for k, v in zip(ids, ref)}
        hypo1 = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref1, hypo=hypo1)
        eval_ce = self.chexbert_metrics.compute(ref, hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)
        self.print(eval_ce)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        current_img_embeds, _,atts_img = self.encode_img(image)
        current_img_embeds = self.layer_norm(current_img_embeds)

        context_image = samples["context_image"]
        context_img_embeds, _,context_atts_img = self.encode_img(context_image)
        context_img_embeds = self.layer_norm(context_img_embeds)
        context_report = samples["context_input_text"]

        current_img_embeds, atts_img = self.prompt_wrap_context(current_img_embeds, atts_img, context_report)

        batch_size = current_img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, current_img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        generated_tokens = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature
        )

        hypo = [self.decode(i) for i in generated_tokens]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref1 = {k: [v] for k, v in zip(ids, ref)}
        hypo1 = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref1, hypo=hypo1)
        eval_ce = self.chexbert_metrics.compute(ref, hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        self.print(f"Test result of {self.hparams.delta_file}: {eval_ce}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
