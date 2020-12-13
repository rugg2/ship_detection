# ship detection & localisation
Ship detection and localisation from satellite images.

**Blog posts:**
- <a href='https://medium.com/@romain.guion/satellite-images-object-detection-part-1-95-accuracy-in-a-few-lines-of-code-8ee4acd72809'>Ship detection - Part 1</a>: Ship detection, i.e. binary prediction of whether there is at least 1 ship, or not. Part 1 is a simple solution showing great results in a few lines of code
- <a href='https://medium.com/@romain.guion/satellite-images-object-detection-part-2-the-beauty-the-beast-f92ff27b696a'>Ship detection - Part 2</a>: ship detection with transfer learning and  decision interpretability through GAP/GMP's implicit localisation properties
- <a href='https://medium.com/vortechsa/satellite-image-segmentation-part-3-eeb134fe3dd5'>Ship localisation / image segmentation - Part 3</a>: identify where ship are within the image, and highlight pixel by pixel

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
    - other data providers: Airbus, Digital Globe
    - free sources: includes EOS's Sentinel 1 (SAR - active/radar) and 2 (optical)  with coverage period ranging of 2-7 days 
        https://eos.com/blog/7-top-free-satellite-imagery-sources-in-2019/
- **simple EDA on data used in this repo and blog post:**
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
    - https://towardsdatascience.com/u-net-b229b32b4a71
    - https://www.tensorflow.org/tutorials/images/segmentation
    - https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    - more classic ship detection algorithms in skimage.segmentation: https://developers.planet.com/tutorials/detect-ships-in-planet-data/
