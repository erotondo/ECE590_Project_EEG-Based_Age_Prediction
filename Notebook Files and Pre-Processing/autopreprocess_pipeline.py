#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:45:43 2019

@author: hannekevandijk

copyright: Research Institute Brainclinics, Brainclinics Foundation, Nijmegen, the Netherlands

"""

### NOTE: NOT MY CODE! Please see the information above!!!

from TD_BRAIN_code.autopreprocessing import dataset as ds
from TD_BRAIN_code.inout import FilepathFinder as FF
import os
from pathlib import Path
import numpy as np
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sourcepath', default=None,
                        help='Path of the original datasets')
    parser.add_argument('--savepath', default=None,
                        help='Folder where the data should be saved')
    parser.add_argument('--condition', default=None,
                        help='Which condition should be preprocessed (EO or EC)')
    parser.add_argument('--subject', default=None,
                        help='Subject-specific IDcode e.g. 12345678')
    parser.add_argument('--startsubj', default=0, type=int,
                        help='Subject to resume with preprocessing')

    config = parser.parse_args()
    return config

def autopreprocess_standard(varargsin, subject=None, startsubj=0):
    """ standard autopreprocessing pipeline
    varargsin is a dictionary required with fields:
        ['sourcepath']: path of the original datasets
        ['savepath']: folder where the data should be saved
        ['condition']: which condition should be preprocessed
    subject: (optional) if a specific subject should be processed, should be IDcode e.g. 12013456
            but can also be can be the nth file in a folder
        """
    # Defining the reading path
    if varargsin['sourcepath'] is None:
        raise ValueError('sourcepath not defined, where is your data?')

    if varargsin['savepath'] is None:
        raise ValueError('savepath not defined')

    sourcepath = os.path.expanduser(varargsin['sourcepath'])
    preprocpath = os.path.expanduser(varargsin['savepath'])

    print(sourcepath)
    print(preprocpath)
    #print(os.path.expanduser(sourcepath))
    #print(os.path.exists(os.path.expanduser(sourcepath)))

    #find all csv files in the 'derivatives' folder
    csv = FF('eeg.csv',sourcepath)
    csv.get_filenames()
    if len(csv.files)<1:
        raise ValueError('no csv files found in this specified path, please check your sourcepath '+sourcepath)

    #other variables
    if varargsin['condition'] is None:
        reqconds = ['EO','EC']
    else:
        reqconds = varargsin['condition']

    if not 'exclude' in varargsin:
        varargsin['exclude'] = []

    rawreport = 'yes'

    #Inventory of all relevant subjects
    subs = [s for s in os.listdir(sourcepath) if os.path.isdir(os.path.join(sourcepath,s)) if not any([e in s for e in ['preprocessed','results','DS']])]
    subs = np.sort(subs)
    print(str(len(subs))+' subjects')
#    subs = [s for s in os.listdir(sourcepath+'/') if os.path.isdir(os.path.join(sourcepath,s)) and not '.' in s]
#    subs = np.sort(subs)
    k=startsubj
    if subject == None:
        subarray = range(k,len(subs))
    elif type(subject) ==int:
        subarray = [subject]
    elif type(subject) == str:
        subarray = np.array([np.where(subs==('sub-'+subject))[0]][0])
    sp = k
    for s in subarray:
        print('[INFO]: processing subject: '+str(sp) +' of '+str(len(subs)))
        sessions = [session for session in os.listdir(os.path.join(sourcepath,subs[s])) if not any([e in session for e in ['preprocessed','results','DS']])]
        subs = np.sort(subs)
        for sess in range(len(sessions)):
            conditions = []
            allconds = np.array([conds for conds in os.listdir(os.path.join(sourcepath,subs[s],sessions[sess]+'/eeg/')) if not any([e in conds for e in ['preprocessed','results','DS']])])
            if reqconds == 'all':
                conditions = allconds
            else:
                conditions = np.array([conds for conds in allconds if any([a.upper() in conds for a in reqconds])])

            for c in range(len(conditions)):
                print(conditions[c])
                if len(conditions)>0:
                    inname = os.path.join(sourcepath, subs[s], sessions[sess]+'/eeg/',conditions[c])
                    #try:
                    tmpdat = ds(inname)
                    tmpdat.loaddata()
                    tmpdat.bipolarEOG()
                    tmpdat.apply_filters()
                    tmpdat.correct_EOG()
                    tmpdat.detect_emg()
                    tmpdat.detect_jumps()
                    tmpdat.detect_kurtosis()
                    tmpdat.detect_extremevoltswing()
                    tmpdat.residual_eyeblinks()
                    tmpdat.define_artifacts()

                    trllength = 'all'#5#'all'
                    npy = copy.deepcopy(tmpdat)
                    npy.segment(trllength = trllength, remove_artifact='yes')#remove_artifact='no')
                    # subpath = os.path.join(preprocpath,subs[s])
                    # Path(subpath).mkdir(parents=True, exist_ok=True)
                    sesspath = os.path.join(preprocpath,subs[s],sessions[sess]+'/eeg/')
                    Path(sesspath).mkdir(parents=True, exist_ok=True)
                    npy.save(sesspath)

                    if rawreport == 'yes':#for the raw data report
                        lengthtrl = 10
                        pdf = copy.deepcopy(tmpdat)
                        pdf.segment(trllength = lengthtrl, remove_artifact='no')
                        pdf.save_pdfs(sesspath)
                   # except:
                   #     print('processing of '+inname+ ' went wrong')
        sp=sp+1

def main():
    config = vars(parse_args())
    autopreprocess_standard(config, config['subject'], config['startsubj'])


if __name__ == '__main__':
    main()