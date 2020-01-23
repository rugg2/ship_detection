# ship_detection
Ship detection and localization from satellite images.

<img src="https://rugg2.github.io/project_files/deepneuralnetworks_image/class_activation_mapping.PNG" alt="Class activation mapping on vessel detection classifier ConvNet" height="475">

Steps taken:
- **find data sets:** 
    - planet API: 
        - needs subscription, but there is a free trial
        - Example usage here: https://medium.com/dataseries/satellite-imagery-analysis-with-python-a06eea5465ea
    - airbus kaggle set (selected for first iteration)
        - https://www.kaggle.com/c/airbus-ship-detection/data
        - to download locally, make sure your connection is stable (29 GB)
            - get API key through your kaggle profile (free), and either save file or enter name and key as environmental variable
            - nohup kaggle competitions download -c airbus-ship-detection & disown %1
    - free sources (unverified quality or recency): 
        https://eos.com/blog/7-top-free-satellite-imagery-sources-in-2019/
- **simple EDA:**
    - 200k+ images of size 768 x 768 x 3
    - 78% of images have no vessel
    - some images have up to 15 vessels
    - Ships within and across images differ in size, and are located in open sea, at docks, marinas, etc.
- **modelling broken down into two steps**, with subfolders in the repo
    - ship detection: binary prediction of whether there is at least 1 ship, or not
    - ship localisation: identify where ship are within the image, and highlight with a bounding box
- **other articles on the topic:**
    - https://www.kaggle.com/iafoss/unet34-dice-0-87/data
    - https://www.kaggle.com/uysimty/ship-detection-using-keras-u-net
    - https://medium.com/dataseries/detecting-ships-in-satellite-imagery-7f0ca04e7964
    - https://github.com/davidtvs/kaggle-airbus-ship-detection
    - https://towardsdatascience.com/deep-learning-for-ship-detection-and-segmentation-71d223aca649
