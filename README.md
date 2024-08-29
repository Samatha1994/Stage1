# Stage1: Model Training and Data Configuration




**Required Inputs:**
1) training data                       (Path: /homes/samatha94/ExAI_inputs_and_outputs/Stage1_Inputs/adkdata/training)
2) validation data                     (Path: /homes/samatha94/ExAI_inputs_and_outputs/Stage1_Inputs/adkdata/validation)
3) owlfile                             (Path: /homes/samatha94/ExAI_inputs_and_outputs/Stage1_Inputs/owlfile/combined.owl)
4) set6-initial_score_hybrid.config    (Path: /homes/samatha94/ExAI_inputs_and_outputs/Stage1_Inputs/set6-initial_score_hybrid.config)


**Expected Outputs:**                  (Path: /homes/samatha94/ExAI_inputs_and_outputs/Stage1_Results)
1) model_resnet50V2_10classes_retest2023June.h5
2) finalClass_with_image_name_retest2023June.csv
3) predictions_of_tenNeurons_dataframe_retest2023June.csv
4) preds_of_64Neurons_denseLayer_1370Images_retest2023June.csv
5) positive_images.csv
6) negative_images.csv
7) config_files/neuron_<neuronid>.config

**Instructions to Set Up the Environment and Run the Python Script:**
1) Install Python 3.11.5

   Ensure Python 3.11.5 is installed on your system by executing below command

   python --version
3) Set Up a Virtual Environment

Install virtualenv if it is not already installed
    
    pip install virtualenv
5) Create a virtual environment named 'venv'

   python -m venv venv
7) Activate the Virtual Environment

   On macOS/Linux:

   source venv/bin/activate

   On Windows:

   venv\Scripts\activate
9) Install Required Python Packages:

   pip install tensorflow Pillow scipy pandas scikit-learn gdown
10) Run the Python Script

   python main.py

**Steps to Run the Script on BeoCat:**

**Bash file name:** job_stage1.sh

**Bash Script:** https://github.com/Samatha1994/Bash_scripts/blob/main/job_stage1.sh

**Bash Command to kick off the job:** sbatch job_stage1.sh

**Bash command to check the status of the job:** 

sacct --format=JobID,JobName,State,ReqMem,MaxRSS,Start,End,TotalCPU,Elapsed,NCPUS,NNodes,NodeList --jobs= <job_id>

**Log file:** my_job_output_<job_id>.txt (Path: /homes/samatha94/)

**Bash Command to cancel the job:** scancel job_id


