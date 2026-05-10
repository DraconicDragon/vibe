# CONTEXT: Every score is normalized across whole danbooru for each scorer
# source: https://github.com/Disty0/dataset-helpers/blob/main/create-tags-from-json.py

danbooru_quality_scores = {
    "g": {6: 50, 5: 30, 4: 20, 3: 10, 2: 5, 1: 1},
    "s": {6: 150, 5: 80, 4: 50, 3: 20, 2: 10, 1: 5},
    "q": {6: 300, 5: 200, 4: 100, 3: 50, 2: 25, 1: 10},
    "e": {6: 420, 5: 280, 4: 180, 3: 100, 2: 50, 1: 25},
}

# this model is tiny <1mb and outputs yes/no
aes_wd14_scores = {
    6: 0.999666,
    5: 0.9983,
    4: 0.992,
    3: 0.50,
    2: 0.016,
    1: 0.0002,
}
aes_shadow2_scores = {
    6: 0.938,
    5: 0.925,
    4: 0.911,
    3: 0.875,
    2: 0.825,
    1: 0.750,
}
aes_deepghs_swinv2pv3_x_scores = {
    6: 0.962,
    5: 0.890,
    4: 0.786,
    3: 0.585,
    2: 0.388,
    1: 0.192,
}
aes_euge3_scores = {
    6: 0.8396,
    5: 0.7405,
    4: 0.6942,
    3: 0.3698,
    2: 0.2940,
    1: 0.1569,
}


quality_score_to_tag = {
    6: "best quality",
    5: "high quality",
    4: "great quality",
    3: "normal quality",
    2: "low quality",
    1: "bad quality",
    0: "worst quality",
}


aes_score_to_tag = {
    6: "very aesthetic",  # less than 1000 images are able to get this score when using multiple aes models
    5: "very aesthetic",
    4: "highly aesthetic",
    3: "moderate aesthetic",
    2: "low aesthetic",
    1: "bad aesthetic",
    0: "worst aesthetic",
}
