# fatigue-driving-detection
[English](README.md) | [简体中文](README.zh-CN.md)

Project for the Challenge Cup 2023 Huawei Industrial Contest: **Fatigue Driving Detection**

Knowledge stacks involved include: Face detection, Facial keypoint recognition, Sequential judgement.

---

## Model Description

For the key models, we've tried the following:

1. YOLOv5 + Dlib
2. YOLOv5 + SPIGA
3. YOLOv7 + SPIGA
4. **YOLOv7 + Renita (baseline)**
5. YOLOv8 + SPIGA
6. **YOLOv8 + Renita (last submit)**
    
    ---
    

## Parameter Tuning Experience

- The **YOLOv5s** model is slightly inferior to **YOLOv8n** in terms of both **accuracy** and **speed**.
  
- The **YOLOv7** model has a training time several times longer than a similar parameter model of v8.

- **SPIGA** is a SOTA model for facial keypoint detection, but obviously, the SOTA is the **accuracy** on large datasets, not **accuracy/performance**. Its average inference speed per frame on the official required hardware (an old 2-core 8GB cpu) reached an astonishing ***1.404s per frame***(*😓). It was eventually eliminated due to its poor edge deployment capability.
> For reference, at 640 x 640
> 
> YOLOv8: 230ms per frame
> 
> Renita: 56ms per frame
> 
- **Renita** is a model reference provided by the official Baseline, and is the model we finally chose. This model performs well in a car interior environment where the occlusion and lighting conditions are not complicated.
  💡 As a reference, the average single frame inference speed on official hardware is:


---

## File Guidance

BaselineLandmark/detectionx.2: The last submission

submit/detection8: The final version of YOLOv8 + SPIGA

![Untitled](README/Untitled.jpeg)

BaselineLandmark/makeup: Debugging script for Retina + YOLOv8

![frame_120.jpg](README/frame_120.jpg)


---

## Trivia

**Why is the main code of the main program all lumped together? Don't you encapsulate at all?**

- The main program, in order to be close to the baseline submission format and facilitate tuning, has not been encapsulated. *Although the structure is almost identical, the main program was not referenced to the baseline at all when writing, so the person who started the porting was scratching his head.*
- And the whole project is **single programmer** and **linear work**, everyone's division of work area is clear enough, so there is no need to consider whether others can understand, everyone **does not have the time and energy**~~nobody reads it except me~~.

> The best division of labor is when one person's work covers all aspects of this part of the work.
> 

**How did you do in the competition?**

The competition evaluates the F1-Score (accuracy) and speed. In the facial keypoint part, by using sequential pattern recognition, we have achieved **all that should be checked.**

**What were the difficulties in the competition?**

- The size of the dataset is a bit exaggerated (2066 * 8s video sequences)
- Huawei Cloud ModelArts' online deployment feature (submission for ranking needs to be done through this method), the document is missing, featuring a complete lack of information. Contestants need to reverse engineer its operation process before making a stable submission 😓.
