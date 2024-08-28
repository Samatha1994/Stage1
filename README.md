Stage1: Model Training and Data Configuration




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


**Bash file name:** job_stage1.sh

**Bash Command to kick off the job:** sbatch job_stage1.sh

**Bash command to check the status of the job:** 

sacct --format=JobID,JobName,State,ReqMem,MaxRSS,Start,End,TotalCPU,Elapsed,NCPUS,NNodes,NodeList --jobs= <mention_your_job_id_here>

**Log file: **	my_job_output_<job_id>.txt (Path: /homes/samatha94/)


