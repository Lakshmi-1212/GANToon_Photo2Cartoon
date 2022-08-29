# GANToon_Photo2Cartoon


**STEPS TO RUN**  

**TRAINING:**  
  
To train the model with default settings  

$python src/train.py 

**TESTING:**  
  
$python src/test_load_model.py   
   
By default, 
- Trained models - save_model folder
- Test folder with the real-world photos - dataset/test/test_photo folder
- Directory to save the output - results/Hayao folder
Parameters can be overridden while running the test application

**POINTS TO NOTE**

1. Dataset should be present under the root directory,   
   GANToon_Photo2Cartoon/dataset   

  
2. Download the pre-trained VGG19 model and place it in the following folder:  
   Destination: GANToon_Photo2Cartoon/src/utils/vgg19.npy


