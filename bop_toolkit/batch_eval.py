import subprocess
import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, default="/workspace/synthetica-vision/")
args = parser.parse_args()

json_dir = args.json_dir

dataset = 'tless'

json_file_name = f'synthetica_{dataset}-test_epoch-id-*.json'

ckpt_jsons = sorted(glob.glob(json_dir + f'{json_file_name}'))

benchmarking_file = open('benchmarking_results.txt', 'w')

for ckpt_json in ckpt_jsons:

    print('benchmarking {}'.format(ckpt_json))

    copied_json = json_dir + f"synthetica_{dataset}-test.json"

    os.system("cp {} {}".format(ckpt_json, copied_json))

    # Define the command to run
    command = f"python scripts/eval_bop22_coco.py --result_filenames={json_dir}/synthetica_{dataset}-test.json --ann_type='bbox'"

    # Run the command and capture the output
    result = subprocess.run(command, shell=True, text=True, capture_output=True)

    # Get the output as a string and split into lines
    output_lines = result.stdout.splitlines()

    #print(output_lines)

    # Extract the last 12 lines
    last_12_lines = output_lines[-12:]

    # Join the last 12 lines back into a single string
    coco_eval_summary = "\n".join(last_12_lines)

    best_mAP = coco_eval_summary.split('\n')[0].rsplit('=')[-1]

    epoch_num = ckpt_json.split('.json')[0].split('synthetica_{}-test_epoch-id-'.format(dataset))[1]
    # Print or process the summary
    benchmarking_file.write('Epoch {}\n\n'.format(epoch_num))
    benchmarking_file.write(coco_eval_summary)
    benchmarking_file.write('\n')
    benchmarking_file.write('\n')

    print(coco_eval_summary)

    print('----------')


benchmarking_file.close()

