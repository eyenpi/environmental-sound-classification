# 🎶 Environmental Sound Classifier
## 📖 Project Overview
This project aims to classify environmental sounds into one of 10 categories. Whether it’s the sound of a dog barking 🐕, a car horn blaring 🚗📢, or even the rhythmic hum of a jackhammer 🚧🔨, our model is here to figure it all out!

## 🎯 Features:

Dataset: The famous UrbanSound8k 📊🔊
- 8,732 labeled sound excerpts (each ≤ 4 seconds)
- 10 classes:
- ❄️ air_conditioner
- 🚨 car_horn
- 👶 children_playing
- 🐕 dog_bark
- 🛠 drilling
- 🚗 engine_idling
- 🔫 gun_shot
- ⚒ jackhammer
- 🚑 siren
- 🎶 street_music

## 🔍 How It Works:

We built a hand-designed CNN 🧠 with a fraction of the parameters used in larger models like ResNet18 but still packs quite the punch! 💥

- Feature Extraction: 🎛 The CNN helps pull meaningful features out of the sound clips.
- Classification: 🏷 Using a few linear layers, the model classifies the sound into one of the 10 categories. Boom! 🎤

## 🏋️‍♀️ Model Strengths:

- Efficient 🏎: Designed with far fewer parameters than ResNet18.
- Lightweight 🎈: Smaller model = Faster predictions!
- Trained on a high-quality urban sound dataset. 🌆🎧

## 🛠 How to Use

- Clone the repo: git clone https://github.com/yourname/sound-classifier.git 👨‍💻
- Install dependencies: pip install -r requirements.txt 🛠
- Run the training script: python train.py 💪
- Classify some sounds! 🔊

## 🚀 Future Plans

- 🔄 Fine-tune the model for improved accuracy.
- 🎶 Add more sound classes.
- 🌍 Train on a larger dataset for world domination! (Just kidding… or are we? 😏)

Feel free to contribute, test, or even just play around with the code. Let’s make some noise! 🎉
