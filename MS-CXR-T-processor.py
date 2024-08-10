import json
import re
import pandas as pd
import os

final_file={}

def clean_reports(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

csv_file = './ms-cxr-t/1.0.0/MS_CXR_T_temporal_image_classification_v1.0.0.csv'
df = pd.read_csv(csv_file)

result_dict = {}

base_folder = './physionet.org/files/mimic-cxr/2.0.0/files'

for index, row in df.iterrows():
    ids = row['subject_id']
    study_id = row['study_id']
    dicom_id = row['dicom_id']
    previous_dicom_id = row['previous_dicom_id']
    dicom_folder = os.path.dirname(dicom_id)
    previous_dicom_folder = os.path.dirname(previous_dicom_id)

    report_file_path = os.path.join(base_folder, dicom_folder + '.txt')
    previous_report_file_path = os.path.join(base_folder, previous_dicom_folder + '.txt')
    
    if os.path.exists(report_file_path):
        with open(report_file_path, 'r') as f:
            report_content = f.read()
    else:
        report_content = 'Report not found'
    
    if os.path.exists(previous_report_file_path):
        with open(previous_report_file_path, 'r') as f:
            previous_report_content = f.read()
    else:
        previous_report_content = 'Previous report not found'
    
    if report_content and previous_report_content:
        result_dict[index] = {
            'id': ids,
            "study_id": study_id,
            "report": clean_reports(report_content),
            "image_path": [f"{dicom_id}.jpg"],
            "context_image": [f"{previous_dicom_id}.jpg"],
            "context_report": clean_reports(previous_report_content)
        }

with open('MS-CXR-T-LRRG.json', 'w') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)