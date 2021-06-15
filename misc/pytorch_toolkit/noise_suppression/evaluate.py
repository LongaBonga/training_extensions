import logging
import os
import time
import re

import torch

import models
from metrics import sisdr
from dataset import AudioFile

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} evaluate'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))


#load model on cpu and evaluate on long file dataset
def evaluate_dir(eval_data_dir, model_dir):

    if not eval_data_dir or not os.path.isdir(eval_data_dir):
        printlog("{} is not folder with evaluation data! Evaluation for {} is skipped".format(eval_data_dir, model_dir))
        return {}

    #create model from dir
    model = models.model_from_dir(model_dir)
    model.eval()

    #create dataset
    file_names = {"clean":{},"noisy":{}}
    for r, d, files in os.walk(eval_data_dir):
        for f in files:
            if f.split('.')[-1] not in ['wav']:
                continue
            m = re.match(".*fileid_(\d+)\..*", f)
            if m is None:
                printlog("file {}/{} skipped because of missed id".format(r,f))
                continue
            t = os.path.split(r)[-1]
            id = m.group(1)
            if f.startswith("synthetic_singing"):
                id = "singing_" + id
            elif f.startswith("synthetic_emotion"):
                id = "emotion_" + id
            else:
                id = "speech_" + id

            if id in file_names[t]:
                printlog("duplicate id", file_names[t][id], os.path.join(r, f))
            else:
                file_names[t][id] = os.path.join(r, f)

    file_ids = [set(v.keys()) for v in file_names.values()]
    file_ids = list(sorted(set.intersection(*file_ids)))

    for i,fi in enumerate(file_ids):
        printlog(i,fi,
                 file_names["clean"][fi].replace(eval_data_dir, ""),
                 file_names["noisy"][fi].replace(eval_data_dir, "") )


    #iterate over dataset
    sisdr_inps_all = []
    sisdr_outs_all = []
    for fi in file_ids:

        #read clean and noisy signals
        x_clean = AudioFile(file_names["clean"][fi]).read_all()
        x_noisy = AudioFile(file_names["noisy"][fi]).read_all()
        assert x_clean.shape[0] == x_noisy.shape[0]

        input_size = model.get_sample_length_ceil(x_clean.shape[0])
        pad = input_size - x_clean.shape[0]
        if pad>0:
            x_clean = torch.nn.functional.pad(x_clean, (pad, 0), mode='constant', value=0)
            x_noisy = torch.nn.functional.pad(x_noisy, (pad, 0), mode='constant', value=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            model_outputs = model(x_noisy.unsqueeze(0))
        y_clean = model_outputs[0][0]
        t1 = time.perf_counter()

        #shift output by delay to align with input
        delay = model.get_sample_ahead()
        y_clean = y_clean[delay:]
        x_clean = x_clean[:-delay]
        x_noisy = x_noisy[:-delay]

        #calc metrics for input and output
        sisdr_inps_all.append(sisdr(x_noisy, x_clean).item())
        sisdr_outs_all.append(sisdr(y_clean, x_clean).item())

        sample_len = input_size/16000
        sample_time = t1-t0
        printlog("{}/{} {:0.2f}s is evaluated by {:0.2f}s x{:0.2f} sisdr_inp,out,diff {:0.2f} {:0.2f} {:0.2f} for {}".format(
            len(sisdr_inps_all), len(file_ids), sample_len,
            sample_time,
            sample_time / sample_len,
            sisdr_inps_all[-1],
            sisdr_outs_all[-1],
            sisdr_outs_all[-1] - sisdr_inps_all[-1],
            fi
        ))

    result = {}
    for t in ["speech", "singing", "emotion", "all"]:

        inps = [v for fi,v in zip(file_ids,sisdr_inps_all) if t in fi or t=="all"]
        outs = [v for fi,v in zip(file_ids,sisdr_outs_all) if t in fi or t=="all"]

        N = len(inps)
        inp = sum(inps)/N
        out = sum(outs)/N
        diff = out - inp

        result[t+"_sisdr_inp"] = inp
        result[t+"_sisdr_out"] = out
        result[t+"_sisdr_diff"] = diff
        result[t+"_sisdr_inp_stddev"] = (sum((s-inp)**2 for s in inps) / N)**0.5
        result[t+"_sisdr_out_stddev"] = (sum((s-out)**2 for s in outs) / N)**0.5
        result[t+"_sisdr_diff_stddev"] = (sum((so-si-diff)**2 for so,si in zip(outs, inps)) / N)**0.5

    with open(os.path.join(model_dir,"_eval.log"),"at") as out:
        for n, v in result.items():
            out.write("{} - {}\n".format(n, v))
            printlog("{} - {}".format(n, v))
        out.write("result {}\n".format(result))
        printlog("result {}".format(result))
