import os
import sys
#command line arguments
import argparse
import numpy as np
import base64

IN_base64 = "/Td6WFoAAATm1rRGAgAhARYAAAB0L+Wj4E//AyRdADKcQiFazHbptEya0uKdmUo0mBgwrstdAmMYQ1xkaobm9GDzI481Xf/ikhJG+1nx8iQDfR3lo1QCENSfMS41I+ztkyTT4hVZkiTZzdTvit+QvhcdSpYlOXxHbbHFfYL2SNsvOoU/yKcykz/6cW/6/XVLVnKpc/PJophMO7BQYv1hmUxZlFPbh1ukjFgD9qkyh8+zl9DAltD1hJy5Gi2YFvSOPvRGd+Wp5g1N+3os+aDSZBxR6OVHs+NYAys81bcPkBd3xYZh1zrGWWdnpT4liFg8xhlcu9sr4Uzfg53D/Fbtp9vN4w+z0SK/QF+tK17eydJxcehZ8egTv/uxaT2pmMWFXPaFM9mXLUebtznSFQPnvy7nXAjxB84yMM19I367qx7qg9s99xrVCjSrre9napRFZmfNaka7bajW+29YpMtadD92oJQWG3O9HLNOSBVw+9Pj/t0+hIK6fwLVrGrZC4Am7qDEiG/yt/GMUGigowy5y9W7z1aaFc6od6CSxDJ+wApQUuJdrCS+o7WEdm8uiMfSMd45+E7nFUuIiqHWQY8O7WOwVXJvtQlhGPt6SsXVT1bKAfTI1G3NRishTIr/f+6XQP0IAi0u9Dx4yXwc38ant0m7OoGTz6pH2osq+BykSnpLpwVL8roZqkb5Oy7DBD6M6rVvmforaefK27gFri1LVcKTffi0LRVR2XOWp3uE/Rw4xOS2jKqTJlGlUqUM6mr4B0cY+kUtDY0MpAcmMkCu5Bl7/+P5YN2fyaD7oUEqtYOTgU4NXnTCDbc9kTn88WPZxDOHUFpj7hmUiqC2IMNuSgoZpyAG8a/XkyKTeQzNyorwHyJbhYPGFq1J1W2ctXg7xLE78SBrz4r3Gt86iHv01KqSrjyylFdzi032zerJTPvTQtNfJMDFRvSCRPbM/OHypPCdtbU0tR8e6jD6pvaZ6/3/YT9d3oQn88IXPbeIRyFdOrb03C+7DyO8fWznMRXvgoSM4pCBzVnE1R+exRXsl/V80/TmPnxYJcR7Sy6X7Xh2XadRMjP/yd2bPCdaLXTRBgWZbNAcDA78AfPBuA4vAo87nQAX0Um/vShpwQABwAaAoAEA0+XHcLHEZ/sCAAAAAARZWg=="

def write_submission_script_body(f):
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i min1.in  -o min1.out   -p *prmtop -c *.inpcrd -r min1.rst   -x min1.nc   -ref *.inpcrd\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i min2.in  -o min2.out   -p *prmtop -c min1.rst -r min2.rst   -x min2.nc   -ref min1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat1.in  -o heat1.out   -p *prmtop -c min2.rst  -r heat1.rst   -x heat1.nc   -ref min2.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat2.in  -o heat2.out   -p *prmtop -c heat1.rst  -r heat2.rst   -x heat2.nc   -ref heat1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat3.in  -o heat3.out   -p *prmtop -c heat2.rst  -r heat3.rst   -x heat3.nc   -ref heat2.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat4.in  -o heat4.out   -p *prmtop -c heat3.rst  -r heat4.rst   -x heat4.nc   -ref heat3.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat5.in  -o heat5.out   -p *prmtop -c heat4.rst  -r heat5.rst   -x heat5.nc   -ref heat4.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i heat6.in  -o heat6.out   -p *prmtop -c heat5.rst  -r heat6.rst   -x heat6.nc   -ref heat5.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i eq1.in  -o eq1.out   -p *prmtop -c heat6.rst  -r eq1.rst   -x eq1.nc   -ref heat6.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i eq2.in  -o eq2.out   -p *prmtop -c eq1.rst  -r eq2.rst   -x eq2.nc   -ref eq1.rst\n')
    f.write(f'/usr/local/amber22/bin/pmemd.cuda -O -i prod.in -o prod.out -p *prmtop -c eq2.rst  -r prod.rst -x prod.nc\n')
    #f.write('#### submit_job.sh END ####\n')

def write_submission_script_header(f,name,gpu):
    f.write('#### submit_job.sh START ####\n')
    f.write('#!/bin/bash\n')
    f.write('#$ -cwd\n')
    f.write(f'#$ -N {name}\n')
    f.write(f'#$ -q {gpu}\n')
    f.write('\n')
    f.write('export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64\n')
    f.write('export AMBERHOME=/usr/local/amber22/\n')
    f.write('\n')

def make_work_dir(name,num_gpu,num_replica,index,index_l):
    #print current working directory
    cwd = os.getcwd()
    print(f'Current working directory: {cwd}')
    #calculate number of replicas per GPU
    num_replica_per_gpu = int(num_replica / num_gpu)
    print(f'Replicas per GPU: {num_replica_per_gpu}')
    #warn if there remain replicas
    if num_replica % num_gpu != 0:
        print(f'Warning: cannot evenly distribute {num_replica} replicas over {num_gpu} GPUs')
    #create a 2D list 
    replica = []
    for i in range(num_gpu):
        replica.append([])
    #create work directory
    for i in range(num_replica):
        #determine which GPU to use
        gpu = int(i / num_replica_per_gpu)
        if gpu >= num_gpu:
            gpu = num_gpu - 1
        #create directory
        dir_name = f'{name}_replica_{i}_gpu_{gpu}'
        print(f'Creating directory: {dir_name}')
        replica[gpu].append(dir_name)
        try:
            os.mkdir(dir_name)
        except OSError:
            #print(f'Creation of the directory {dir_name} failed')
            pass
        #copy files
        os.system(f'cp *.prmtop {dir_name}')
        os.system(f'cp *.inpcrd {dir_name}')
        os.chdir(dir_name)
        #write input files
        with open("in.tar.xz", "wb") as f:
            f.write(base64.b64decode(IN_base64))
        os.system('tar -xf in.tar.xz')
        modifiy_input(index,index_l)
        os.system('rm in.tar.xz')
        os.chdir("..")
    return replica

def write_submission_script(name,replica,gpu):
    #create submission script
    submission_script_list = []
    for gpu_id in range(len(replica)):
        submission_script_list.append(f'submit_job_gpu_{gpu_id}.sh')    
        with open(f'submit_job_gpu_{gpu_id}.sh','w') as f:
            write_submission_script_header(f,f'{name}_gpu_{gpu_id}',gpu)
            for dir_name in replica[gpu_id]:
                f.write(f'cd {dir_name}\n')
                write_submission_script_body(f)
                f.write(f'cd ..\n')
            f.write('#### submit_job.sh END ####\n')
        os.system(f'chmod +x submit_job_gpu_{gpu_id}.sh')
    return submission_script_list

def modifiy_input(index,index_l):
    #list all the .in files in the current directory
    in_files = [f for f in os.listdir('.') if f.endswith('.in')]
    #modify the input files
    for i in in_files:
        #print(f'Modifying {i}')
        content = ""
        with open(i,'r') as f:
            content = f.read()
        #replace the index
        content = content.replace("[index]",index)
        content = content.replace("[index_l]",index_l)
        with open(i,'w') as f:
            f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create submission script for AMBER simulation')
    parser.add_argument('--name', type=str, help='job name', default='mmgbsa')
    parser.add_argument('--num_gpu', type=int, help='number of GPUs', default=4)
    parser.add_argument('--num_replica', type=int, help='number of replicas', default=20)
    parser.add_argument('--gpu', type=str, help='GPU queue', default='gtx2080ti_gpu.q')
    parser.add_argument('--index', type=str, help='index of the last protein residue', default='281')
    parser.add_argument('--index_l', type=str, help='index of the ligand', default='282')
    parser.add_argument('--submit', type=bool, help='If submit', default=False)
    args = parser.parse_args()
    replica = make_work_dir(args.name,args.num_gpu,args.num_replica,args.index,args.index_l)
    submission_list = write_submission_script(args.name,replica,args.gpu)
    if args.submit:
        for i in submission_list:
            os.system(f'qsub {i}')
    else:
        print('Not submitting jobs')


   

