# The Lock Assistor

My Video: https://drive.google.com/file/d/1EEOGpbfONp7J3XTBogVYoyRHTnLs5hOU/view?usp=sharing

My Project Planning Worksheet: https://drive.google.com/file/d/1ORE0L6ttKudealAljKqKAY1Hk8wc1wQG/view?usp=sharing

# Overview
    My project is mostly made for people who loose their keys or forget the combination to their locks and don't know how to access their belongings. My project solves this problem by determining the type of lock based on the image uploaded and then directing the user towards a video or a number to call with the output being different based on the type of lock. The type of locks were narrowed down to combination locks and key locks.


## The Algorithm

    What the algorithm needs to do is determine if an image is either a key lock or a combination lock. Therefore, the algorithm obviously needs a photo of a keylock or a combination lock, but it also needs a file (labels.txt) that gives the algorithm two output options (combination lock or key lock). Finally the algorithm needs a model that can distinguish the difference between a key lock and a combination lock. This model was created by finding hundreds of photos and seperating them into 2 categories (key lock and combination lock). Then the AI is trained to recognize patterns in the shapes of the combination locks and the keylocks. Finally, the trained AI model is exported and put into the code to be used. 
    
    The code is able to use this model because the path (where the model is) is pasted into the code so that the code knows where to find the model. This also is done for the image path and labels path. Since the code has access to all 3 of the necessary parts it is able to run the algorithhm. This is done by having the image be compared to massive amounts of photos in the model known as resnet18.onnx. However, the the model has already been trained so it is compared to the sum of patterns that the model believes a key lock or combination lock should look like. Should the model come to the conclusion that the image is a key lock or a combination lock it changes a variable (the output) to be as such. Finally, if that variable is considered a key lock it prints a certain statement in the termianl (the final output) that gives the user information about the type of lock and how to unlock it. Furthermore, if that variable turns out to be considered a combination lock, a different statement is printed that gives the user information about the type of lock and how to unlock it.

## Running this project

1. Make sure that you can use the onnx, numpy, and PIL library. If you can't, install them by typing "pip instal" and then the name of the library you want to instal.
2. Drag and drop your photos of combination or key locks or copy and paste them next to your final code under the EXPLORER. 
3. Right click on the photo and click copy path.
4. In the code under finalCode.py paste in the path under image_path between the quotation marks.
5. Run the code by clicking the play button in the top right corner and see your output in the terminal below.

