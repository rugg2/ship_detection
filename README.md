# ship_detection
Ship detection and localization from satellite images

Notes for self, to be wrapped up in a blog post.

Steps taken:
- find data set: work on airbus kaggle set
    - https://www.kaggle.com/c/airbus-ship-detection/data
    - to download locally, make sure your connection is stable (29 GB)
        - nohup kaggle competitions download -c airbus-ship-detection & disown %1
- simple EDA: 
    - 200k+ images of size 768 x 768 x 3
    - 77% of images have no vessel
    - some images have up to 15 vessels
    - Ships within and across images differ in size, and are located in open sea, at docks, marinas, etc.
- modelling broken down into two steps, with subfolders in the repo
    - ship detection: binary prediction of whether there is at least 1 ship, or not
    - ship localisation: identify where ship are within the image, and highlight with a bounding box