import cv2
import torch
from PIL import Image

#train parameters for best.pt model
#python train.py --img 640 --batch 32 --epochs 100 --data data.yaml --weights yolov5s.pt
#check https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#3-select-a-model

#load model(enter the path to the best.pt file)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Your_path_to_the_best.pt', force_reload=True)

class_names= [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel',
    'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
    'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound',
    'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
    'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
    'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier',
    'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
    'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
    'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont',
    'Boston_bull', 'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer',
    'Scotch_terrier', 'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
    'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
    'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
    'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
    'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
    'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke',
    'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
    'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
    'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog',
    'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
    'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
    'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
    'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond',
    'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
]

model.eval()

def detect(frame):
    img = Image.fromarray(frame)
    results = model(img)
    results = results.xyxy[0].numpy()

    for result in results:
        x1, y1, x2, y2, conf, cls = result
        cls_id = int(cls)
        if conf > 0.25:
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            confidence_text = f'{class_names[cls_id]}: {conf*100:.2f}%'
            cv2.putText(frame, confidence_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

#choose video source(webcam is usually mark as 0 if it isn't, edit and enter next number 1, 2, 3 etc)
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect(frame)

    cv2.imshow('YOLOv5 Dog Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
