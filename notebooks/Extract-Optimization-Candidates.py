#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys
import tqdm
import time
import numpy as np
import pickle
import random
import yaml
import threading
import time
import datetime

from collections import defaultdict
from yacos.essential import Engine, IO
from yacos.info import compy
from yacos.info.compy.extractors import LLVMDriver


# In[2]:


# Instantiate the LLVM driver.
driver = LLVMDriver()
# Define the builder
builder = compy.LLVMGraphBuilder(driver)


# In[3]:


visitors = {
    # Clang
    'ast': compy.ASTVisitor,
    'astdata': compy.ASTDataVisitor,
    'astdatacfg': compy.ASTDataCFGVisitor,
    # LLVM
    'programl': compy.LLVMProGraMLVisitor,
    'programlnoroot': compy.LLVMProGraMLNoRootVisitor,
    'cfg': compy.LLVMCFGVisitor,
    'cfgcompact': compy.LLVMCFGCompactVisitor,
    'cfgcall': compy.LLVMCFGCallVisitor,
    'cfgcallnoroot': compy.LLVMCFGCallNoRootVisitor,
    'cfgcallcompact': compy.LLVMCFGCallCompactVisitor,
    'cfgcallcompact1e': compy.LLVMCFGCallCompactOneEdgeVisitor,
    'cfgcallcompactnoroot': compy.LLVMCFGCallCompactNoRootVisitor,
    'cfgcallcompact1enoroot': compy.LLVMCFGCallCompactOneEdgeNoRootVisitor,
    'cdfg': compy.LLVMCDFGVisitor,
    'cdfgcompact': compy.LLVMCDFGCompactVisitor,
    'cdfgcompact1e': compy.LLVMCDFGCompactOneEdgeVisitor,
    'cdfgcall': compy.LLVMCDFGCallVisitor,
    'cdfgcallnoroot': compy.LLVMCDFGCallNoRootVisitor,
    'cdfgcallcompact': compy.LLVMCDFGCallCompactVisitor,
    'cdfgcallcompact1e': compy.LLVMCDFGCallCompactOneEdgeVisitor,
    'cdfgcallcompactnoroot': compy.LLVMCDFGCallCompactNoRootVisitor,
    'cdfgcallcompact1enoroot': compy.LLVMCDFGCallCompactOneEdgeNoRootVisitor,
    'cdfgplus': compy.LLVMCDFGPlusVisitor,
    'cdfgplusnoroot': compy.LLVMCDFGPlusNoRootVisitor
}


# In[4]:


dataset_dir = '/home/nonroot/experiment/datasets/classify_seqs/group1.1000/'
folders = glob.glob(f"{dataset_dir}/*/*/")
print(f"There are {len(folders)} directories")


# In[5]:


results_dir = '/home/nonroot/experiment/results/notebook/'
os.makedirs(results_dir, exist_ok=True)


# In[6]:


def extract_graph_data(graph, graph_type):
    """Convert the graph to StellarGraph representation."""

    nodes = {}

    if 'ast' in graph_type:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ast')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ast')
    elif 'asm' in graph_type:
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ir')
    else:
        nodes['w2v'] = graph.get_nodes_word2vec_embeddings('ir')
        nodes['boo'] = graph.get_nodes_bag_of_words_embeddings('ir')
        nodes['i2v'] = graph.get_nodes_inst2vec_embeddings()
        nodes['ir2v'] = graph.get_nodes_ir2vec_embeddings()

    edges = graph.get_edges_dataFrame()

    return edges, nodes


# In[7]:


def compile_and_extract_info(benchdir, optname, optseq):
    d = dict()
    try:
        Engine.cleanup(benchdir, 'opt')
        Engine.compile(benchdir, 'opt', optseq)
        binsize = IO.load_yaml(os.path.join(benchdir, 'binary_size.yaml'))
        codesize = IO.load_yaml(os.path.join(benchdir, 'code_size.yaml'))
        compiletime = IO.load_yaml(os.path.join(benchdir, 'compile_time.yaml'))       
        d.update(binsize)
        d.update(codesize)
        d.update(compiletime)
        return d
    except BaseException as e:
        print(f"Error compiling benchmark at {benchdir} with {optname}: {e}")
        return None
    
    
def compile_bench_with_candidates(benchdir: str, opt: list, optname: str, output_prefix: str, results_dir: str):
    # benchdir = random.choice(folders)
    # optname = random.choice(list(optimization_seqs.keys()))
    class_results = []
    #print(f"Processing: {optname}....")
        
    # Last one will be without any additional candidate (original)
    for i, cand in enumerate(candidates):
        new_opt = opt + cand
        new_opt_str = ' '.join(new_opt)
        new_optname = f'{optname}:{"+".join(cand)}'
        result = compile_and_extract_info(benchdir, new_optname, new_opt_str)
        if not result:
            print(f"Discarding {benchdir} with optseq={new_opt_str} and (optname={optname})")
            return None
        
        result['opt'] = new_opt
        result['optname'] = new_optname
        class_results.append(result)
    
    # Compile with no additional candidates
    opt_str = ' '.join(opt)
    result = compile_and_extract_info(benchdir, optname, opt_str)
    if not result:
        print(f"Discarding {benchdir} with optseq={new_opt_str} and (optname={optname})")
        return None
    
    # Let's save the class results
    output_filename = os.path.join(results_dir, f"{output_prefix}_classes.yaml")
    IO.dump_yaml(class_results, output_filename)
    #print(f"Saved classes to {output_filename}")
    
    # Now lets generate the representation!
    try:
        source = f'{benchdir}/a.out_o.bc'
        extinfo = builder.ir_to_info(source)
    except BaseException as e:
        print(f"Error extracting IR from {source}: {e}")
        return None
            
    for rep in representations:
        try:
            graph = builder.info_to_representation(extinfo, visitors[rep])
            edges, nodes = extract_graph_data(graph, rep)
            for extname, extdata in nodes.items():
                output_filename = os.path.join(results_dir, f"{output_prefix}_{extname}")     
                np.savez_compressed(output_filename, edges=edges, nodes=extdata, labels=result)
            #print(f"Representation {rep} extracted for {benchdir}")
        except BaseException as e:
            print(f"Error extracting representation {rep} for {benchdir}: {e}")
            continue
            
    return class_results


# In[1]:


optimization_seqs = {
    '500-bSum':  [
        '-mem2reg', '-jump-threading', '-instcombine', '-early-cse-memssa', '-jump-threading', '-licm', '-early-cse-memssa', 
        '-sroa', '-simplifycfg', '-reassociate', '-instcombine', '-slp-vectorizer', '-early-cse-memssa'
    ],
    '4372-bMed': [
        '-mem2reg', '-early-cse-memssa', '-correlated-propagation', '-instcombine', '-reassociate', '-simplifycfg', 
        '-early-cse-memssa', '-instcombine', '-licm', '-jump-threading', '-simplifycfg', '-dse', '-reassociate', 
        '-early-cse-memssa', '-instcombine'
    ],
    '5624-bLim': [
        '-sroa', '-early-cse-memssa', '-reassociate', '-instcombine', '-simplifycfg', '-licm', '-speculative-execution', 
        '-jump-threading', '-early-cse-memssa', '-simplifycfg', '-instcombine', '-simplifycfg'
    ],
    '6310-bGeo': [
        '-loop-vectorize', '-sroa', '-gvn', '-instcombine', '-simplifycfg', '-instcombine', '-licm', '-gvn', 
        '-correlated-propagation', '-jump-threading', '-mldst-motion', '-early-cse-memssa', '-instcombine', 
        '-simplifycfg' '-instsimplify'
    ],
    '4211-bCap': [
        '-loop-rotate', '-sroa', '-correlated-propagation', '-indvars', '-gvn', '-tailcallelim', '-instcombine', 
        '-jump-threading', '-reassociate', '-instcombine', '-early-cse-memssa'
    ]
}

candidates = [
    ['-simplifycfg'],
    ['-instcombine'],
    ['-early-cse-memssa'],
    ['-gvn'],
    ['-sroa'],
    #['-jump-threading'],
    #['-mem2reg'],
    #['-licm'],
]

representations = [
    'cfgcompact',
    #'cfgcallcompact',
    # 'cfgcallcompact1e',
    'cfgcallcompactnoroot',
    'cfgcallcompact1enoroot',
   # 'cdfgcompact',
   # 'cdfgcompact1e',
   # 'cdfgcallcompact',
   # 'cdfgcallcompact1e',
   # 'cdfgcallcompactnoroot',
   # 'cdfgcallcompact1enoroot'
]

class_programs = 5000
label = 'binary_size'
min_passes = 0
tolerance = 1000000
equals = 0
totals = 0
exhausteds = 0
class_counts = {i: 0 for i, _ in enumerate(candidates)}
not_oks = 0
exists = 0

exp_id = '5opts-5000-2'
final_dir = os.path.join(results_dir, exp_id)
os.makedirs(final_dir, exist_ok=True)
ok_files = []

threads_running = True

def monitoring(interval: float = 30.0):
    global threads_running, class_programs, class_counts, not_oks, equals, exhausteds, tolerance, totals, candidates, exists
    while threads_running:
        oks = sum(list(class_counts.values()))
        max_oks = class_programs*len(candidates)
        samples_per_class = {' '.join(candidates[k]): v for k, v in class_counts.items()}
        output_dict = {
            'Number of sampes per class': {
                ' '.join(candidates[k]): v
                for k, v in class_counts.items()
            },
            '[DISCARDED] Total': not_oks,
            '[DISCARDED] Samples with repeated minimum sizes': equals,
            '[DISCARDED] Compiled the app with same optimization passes': exists,
            '[DISCARDED] With maximum number of samples': exhausteds,
            'Total tested so far': totals,
            'Total tested so far (%)': (oks/max_oks)*100,
            'Tolerance': tolerance,
            'Tolerance (%)': (totals/tolerance)*100,
        }
        output_dict_str = yaml.dump(output_dict, indent=4, default_flow_style=False)
        print(f"[{str(datetime.datetime.now())}] " +
              f"Samples per class: {samples_per_class}; Repeated minimuns: {equals}; Repeated opts: {exists}; Samples exhausted: {exhausteds}; " +
              f"Total: {totals} ({(oks/max_oks)*100:.3f}); Tolerance: {tolerance} ({(totals/tolerance)*100:.3f})")
        time.sleep(interval)

def run():
    global threads_running, class_programs, label, min_passes, tolerance, class_counts, not_oks, exp_id, final_dir, equals, exhausteds, totals, exists, ok_files
    print(f"Started thread: {threading.current_thread().name}....")
    while threads_running:
        #print(f"COUNTS: {class_counts}")
        if not_oks > tolerance:
            print("Tolerance reached!")
            print(f"Classes BEST: {class_counts}")
            threads_running = False
            break    

        finished = True
        for count in class_counts.values():
            if count < class_programs:
                finished = False
                break

        if finished:
            print(f"Finished. Counts: {class_counts}")
            threads_running = False
            break

        benchdir = random.choice(folders)
        optname = random.choice(list(optimization_seqs.keys()))
        outfilename = benchdir[len(dataset_dir):-1].replace('/', '.')
        cut_index = random.choice(range(min_passes, len(optimization_seqs[optname])))
        opt = optimization_seqs[optname][:cut_index]
        optname = f"{optname}:{cut_index}"
        output_prefix = f"{outfilename}_{optname}"
        if os.path.exists(os.path.join(final_dir, f"{output_prefix}_classes.yaml")):
            print(f"Skipping {output_prefix}... Already exists.")
            exists += 1
            not_oks += 1
            continue

        class_results = compile_bench_with_candidates(benchdir, opt, optname, output_prefix, final_dir)
        if not class_results:
            not_oks += 1
            continue
        results = [r[label] for r in class_results]
        totals += 1
        min_label = min(results)
        min_label_indexes = [r==min_label for r in results]
        # print(f"RESULTS: {results}, min_label: {min_label}, min_label_indexes: {min_label_indexes}")
        if len([r for r in min_label_indexes if r]) > 1:
            # print(f"There are repeated minumums ({len(min_label_indexes)})! Skipping")
            equals += 1
            not_oks += 1
            # may remove files...
            continue

        best_index = min_label_indexes.index(True)
        class_counts[best_index] += 1   
        if class_counts[best_index] > class_programs:
            # print(f"Class {best_index} is exahusted ({class_counts[best_index]})! Skipping")
            exhausteds += 1
            not_oks += 1
            continue

        ok_files.append(output_prefix)
        # print(f"OK ({best_index})")

num_workers = 7
start = time.time()
threads = [threading.Thread(target=run, name=f'Extractor {i}') for i in range(num_workers)]
threads.append(threading.Thread(target=monitoring, name='Monitor'))
for t in threads:
    t.start()
for t in threads:
    t.join()
end = time.time()

IO.dump_yaml(ok_files, os.path.join(final_dir, "info.yaml"))
print(f"CLASS Counts: {class_counts}")
print(f"It took {end-start} seconds...")


# In[ ]:




