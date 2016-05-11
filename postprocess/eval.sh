#!/bin/bash
#
#SBATCH -t 3-12:00 # Runtime
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH -o runs/run.out
#SBATCH -e runs/run.err
#SBATCH --mail-type=end
#SBATCH --mail-user=alexwang@college.harvard.edu

# Args 1,2,3,4 should be parameters to gen_preds
# Arg 5 is the name of the folder to store all the files
# Note that arg 5 is flushed (deleted) before being filled
# so make sure it's not a directory you are actually using

#cd $1 # path to directory containing prepare4rouge and ROUGE.pl
echo Deleting old folders...
rm -rf "$5" # maybe overkill; BE CAREFUL THIS WILL DESTROY A LOT!!!
#rm -rf "$5/tmp_GOLD"
#rm -rf "$5/tmp_SYSTEM"
#rm -rf "$5/tmp_OUTPUT"
mkdir -p "$5/tmp_GOLD" # may want to get rid of $1/
mkdir -p "$5/tmp_SYSTEM"

echo Generating predictions and gold...
python gen_preds.py --srctxt $1 --srcfile $2 --predfile $3 --goldfile $4 --outfile $5

echo Preparing for ROUGE...
perl prepare4rouge-simple.pl "$5/tmp_OUTPUT" "$5/tmp_SYSTEM" "$5/tmp_GOLD" # need to modify to accept commandline arguments; switch for generating GOLD

cd "$5/tmp_OUTPUT"
echo Computing ROUGE-L...
perl ../../RELEASE-1.5.5/ROUGE-1.5.5.pl -m -w 1.2 -e ../../RELEASE-1.5.5/data -a settings.xml # ROUGE-L

echo Computing ROUGE-1...
perl ../../RELEASE-1.5.5/ROUGE-1.5.5.pl -m -n 1 -w 1.2 -e ../../RELEASE-1.5.5/data -a settings.xml # ROUGE-1

echo Computing ROUGE-2...
perl ../../RELEASE-1.5.5/ROUGE-1.5.5.pl -m -n 2 -w 1.2 -e ../../RELEASE-1.5.5/data -a settings.xml # ROUGE-2
