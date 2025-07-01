import subprocess

def run_cd_hit(infile, outfile, cutoff, memory):
    '''
    Run a specific cd-hit command
    '''
    # get the right word size for the cutoff
    if cutoff < 0.5:
        word = 2
    elif cutoff < 0.6:
        word = 3
    elif cutoff < 0.7:
        word = 4
    else:
        word = 5

    mycmd = '%s -i %s -o %s -c %s -n %s -T 1 -M %s -d 0' % ('cd-hit', infile, outfile, cutoff, word, memory)
    print(mycmd)
    process = subprocess.Popen(mycmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

from os.path import *

def cluster_all_levels(start_folder, cluster_folder, filename):
    '''
    Run cd-hit on fasta file to cluster on all levels
    '''
    mem = 2000  # memory in mb

    cutoff = 0.8
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile, cutoff=cutoff, memory=mem)

    cutoff = 0.7
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.6
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.5
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)

    cutoff = 0.4
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.3
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem)
        
    cutoff = 0.2
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem) 
        
    cutoff = 0.1
    outfile = join(cluster_folder, '%s_clustered_sequences_%s.fasta' % (filename, str(int(cutoff * 100))))
    if not exists(outfile):
        run_cd_hit(infile=join(start_folder, '%s.fasta' % filename), outfile=outfile,
                   cutoff=cutoff, memory=mem) 
        

if __name__ == "__main__":
    import os

    cluster_folder = "data/clusters"
    start_folder = cluster_folder
    print(os.getcwd())
    print(os.path.exists(start_folder))
    cluster_all_levels(start_folder, 
                    cluster_folder, 
                    filename='all_sequences')
