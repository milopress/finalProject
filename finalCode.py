import onnxruntime as ort
import numpy as np
from PIL import Image

#load ONNX model
model_path = "/home/nvidia/finalProject/models/resnet18_v3.onnx"
session = ort.InferenceSession(model_path)

#get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

#load/process images
image_path = "/home/nvidia/finalProject/com1.png"
image = Image.open(image_path)


#resize image to match model input size
img_size = (224, 224)
image = image.resize(img_size)
image.save("comb3_output.png")

#convert to RGB if needed
if image.mode != 'RGB':
    image = image.convert('RGB')


#convert to numpy array and normalize
image_array = np.array(image).astype(np.float32)
# image_array = image_array / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
image_array = (image_array / 255.0 - mean) / std

image_array = np.transpose(image_array, (2 ,0 ,1))
input_data = np.expand_dims(image_array, axis = 0)

input_data = input_data.astype(np.float32)

#load labels from file
labels_file = "/home/nvidia/finalProject/models/labels.txt"
with open(labels_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

    #run inferences
outputs = session.run(None, {input_name: input_data})
predictions = outputs[0]

#get the predicted class
predicted_class = np.argmax(predictions)

#get the label name
if predicted_class < len(labels):
    label_name = labels[predicted_class]
    print(f"Predictions: {label_name}")

    #print different messages based on the label
    if predicted_class == 0:
        print(f"This is {label_name}")
    elif predicted_class == 1:
        print(f"This is {label_name}")
    else:
        print(f"Detected: {label_name}")

else:
    print(f"Unknown class: {predicted_class}")


#optional confidence scores and details
print(f"Confidence: {predictions[0][predicted_class]:.2f}")
print(f"All scores: {predictions[0]}")

if label_name == "kL":
    print("Watch this video to open the lock yourself (copy and paste the url into your browser): ")
    print("")
    print("https://www.google.com/search?client=firefox-b-1-d&sca_esv=d50a20c434f29bec&sxsrf=AE3TifMCofyk9P2ZWQuIjOfU41iSTSPbdg:1752691442981&q=how+to+open+key+locks&udm=7&fbs=AIIjpHxU7SXXniUZfeShr2fp4giZud1z6kQpMfoEdCJxnpm_3W-pLdZZVzNY_L9_ftx08kwv-_tUbRt8pOUS8_MjaceHuSAD6YvWZ0rfFzwmtmaBgLepZn2IJkVH-w3cPU5sPVz9l1Pp06apNShUnFfpGUJOF8p91U6HxH3ukND0OVTTVy0CGuHNdViLZqynGb0mLSRGeGVO46qnJ_2yk3F0uV6R6BW9rQ&sa=X&ved=2ahUKEwi0m_C2hMKOAxV2k1YBHSa3N9QQtKgLKAF6BAgZEAE&biw=1707&bih=914&dpr=1.5#fpstate=ive&vld=cid:7b265724,vid:1LQoi8unCa4,st:0")
    print("")
    print("If you are unable to open the lock yourself call this number: (206) 736-9303")
if label_name == "cL":
    print("Call this number: (206) 736-9303. This will contact a locksmith because these combination locks are very difficult to open by yourself. But if you want to put in the time to open the lock yourself watch this video by copying and pasting it into your browser:")
    print("")
    print("https://www.google.com/search?q=how+to+pick+combination+locks&client=firefox-b-1-d&sca_esv=d50a20c434f29bec&udm=7&biw=1707&bih=914&sxsrf=AE3TifO6orxGahaZjt-9ER88V3Wxjq_r9A%3A1752691782093&ei=RvR3aJe4BZ_k2roPmaPlgAk&ved=0ahUKEwiX88nYhcKOAxUfslYBHZlRGZAQ4dUDCBA&uact=5&oq=how+to+pick+combination+locks&gs_lp=EhZnd3Mtd2l6LW1vZGVsZXNzLXZpZGVvIh1ob3cgdG8gcGljayBjb21iaW5hdGlvbiBsb2NrczILEAAYgAQYkQIYigUyCxAAGIAEGIYDGIoFMgsQABiABBiGAxiKBTIFEAAY7wVI4RRQrQhY6RJwA3gAkAEAmAHHAaAB2QiqAQMwLja4AQPIAQD4AQGYAgegAu0FwgIKEAAYgAQYQxiKBcICBBAAGB7CAgYQABgIGB7CAggQABiABBiiBMICBxAAGIAEGA3CAgYQABgHGB7CAgYQABgNGB7CAggQABgIGA0YHpgDAIgGAZIHAzMuNKAHhCSyBwMwLjS4B-IFwgcFMC42LjHIBw4&sclient=gws-wiz-modeless-video#fpstate=ive&vld=cid:ed06ec86,vid:UGDM_lsM2B4,st:0")
