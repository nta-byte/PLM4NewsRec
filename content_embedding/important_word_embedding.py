from vncorenlp import VnCoreNLP
import json
import numpy as np
from typing import Text, Dict, List
import re
import os
import math
import pandas as pd

try:
    import config
except Exception as e:
    from . import config

cur_path = os.path.dirname(os.path.abspath(__file__))
important_word_file = os.path.join(cur_path, "data/important_word.json")

annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)

with open(important_word_file, "r", encoding="utf8") as f:
    important_words = json.loads(f.read())

word_count_offset = 0.001


def post_processing(text: Text):
    text = text.strip()
    text = text.replace("òa", "oà")
    text = text.replace("óa", "oá")
    text = text.replace("ỏa", "oả")
    text = text.replace("õa", "oã")
    text = text.replace("ọa", "oạ")
    text = text.replace("òe", "oè")
    text = text.replace("óe", "oé")
    text = text.replace("ỏe", "oẻ")
    text = text.replace("õe", "oẽ")
    text = text.replace("ọe", "oẹ")
    text = text.replace("ùy", "uỳ")
    text = text.replace("úy", "uý")
    text = text.replace("ủy", "uỷ")
    text = text.replace("ũy", "uỹ")
    text = text.replace("ụy", "uỵ")
    text = text.replace("Ủy", "Uỷ")
    ret = re.sub(r"\s+", " ", text)
    ret = ret.lower()
    return ret


def tokenize(text: Text):
    word_segmented_text = annotator.tokenize(text)
    return [word for sent in word_segmented_text for word in sent]


def get_all_destination():
    ret = []
    for destination in important_words[config.TagGroups.DESTINATION]:
        destination = post_processing(destination)
        ret.append(destination)

    return ret


def get_all_departure():
    ret = []
    for destination in important_words[config.TagGroups.DEPARTURE]:
        destination = post_processing(destination)
        ret.append(destination)

    return ret


def get_destination_brand_from_resort(text: Text, word_cnt):
    destinations = dict()
    brands = dict()
    for tag in important_words[config.TagGroups.RESORT]:
        destination = post_processing(important_words[config.TagGroups.RESORT][tag][config.TagGroups.DESTINATION])
        brand = post_processing(important_words[config.TagGroups.RESORT][tag][config.TagGroups.BRAND])
        tag_p = post_processing(tag)
        tag_cnt = text.count(tag_p)
        if tag_cnt:
            score = math.sqrt(tag_cnt / word_cnt)
            destinations[destination] = score
            brands[brand] = score
    return destinations, brands


def get_destination_departure_score(text: Text, all_destinations, all_departures, word_cnt):
    ret = []
    ret_destinations, ret_brands = get_destination_brand_from_resort(text, word_cnt)

    ret_departures = dict()

    if len(ret_destinations) == 0:
        for destination in all_destinations:
            destination_cnt = text.count(destination)
            score = math.sqrt(destination_cnt / word_cnt)
            if destination_cnt > 0:
                ret_destinations[destination] = score

    for departure in all_departures:
        if departure not in ret_destinations:
            destination_cnt = text.count(departure)
            score = math.sqrt(destination_cnt / word_cnt)
            if destination_cnt > 0:
                ret_departures[departure] = score

    for departure in all_departures:
        ret.append(ret_departures.get(departure, 0.0))
    for destination in all_destinations:
        ret.append(ret_destinations.get(destination, 0.0))
    return ret


def get_length_of_stay_score(text: Text, word_cnt):
    def add_to_nights_cnt(_nights_cnt, _night):
        if _night >= 0:
            if _night > config.MAX_NIGHTS:
                _night = config.MAX_NIGHTS
            if _night in _nights_cnt:
                _nights_cnt[_night] += 1
            else:
                _nights_cnt[_night] = 1

    nights_cnt = dict()
    ret = []
    day_pattern = r"\d+\s*ngày"
    night_pattern = r"\d+\s*đêm"
    for match in re.finditer(day_pattern, text):
        day = match.group()[:-4]
        try:
            night = int(day) - 1
            add_to_nights_cnt(nights_cnt, night)
        except Exception as e:
            pass

    for match in re.finditer(night_pattern, text):
        day = match.group()[:-3]
        try:
            night = int(day)
            add_to_nights_cnt(nights_cnt, night)
        except Exception as e:
            pass
    for it in range(config.MAX_NIGHTS + 1):
        night_cnt = nights_cnt.get(it, 0)
        score = math.sqrt(night_cnt / word_cnt)
        ret.append(score)
    return ret


def get_all_tag_group_score(text: Text, word_cnt, special_tag_group):
    ret = []
    for tag_group in important_words:
        if tag_group not in special_tag_group:
            for tag in important_words[tag_group]:
                tags = [post_processing(tag)]
                if isinstance(important_words[tag_group], Dict) \
                        and isinstance(important_words[tag_group][tag], Dict) \
                        and config.SIMILAR_WORD_KEY in important_words[tag_group][tag]:
                    other_tags = important_words[tag_group][tag][config.SIMILAR_WORD_KEY]
                    if isinstance(other_tags, List):
                        for tmp in other_tags:
                            tags.append(post_processing(tmp))
                tag_cnt = 0
                for t in tags:
                    tag_p = post_processing(t)
                    tag_cnt += text.count(tag_p)
                score = math.sqrt(tag_cnt / word_cnt)
                ret.append(score)
    return ret


def embedding(text: Text):
    # get word count vncorenlp
    # word_cnt = get_word_count_vncorenlp(text)
    # text = post_processing(text)
    # get word count faster
    text = post_processing(text)
    word_cnt = get_word_count(text)
    # end of get word count
    all_destinations = get_all_destination()
    all_departures = get_all_departure()

    special_tag_group = set()
    ret = []
    # Get Departure and Destination first
    special_tag_group.add(config.TagGroups.DEPARTURE)
    special_tag_group.add(config.TagGroups.DESTINATION)
    ret.extend(get_destination_departure_score(text, all_destinations, all_departures, word_cnt))
    # Get Length of stay following
    ret.extend(get_length_of_stay_score(text, word_cnt))
    special_tag_group.add(config.TagGroups.LENGTH_OF_STAY)
    # Get others
    ret.extend(get_all_tag_group_score(text, word_cnt, special_tag_group))

    return np.array(ret)


def get_word_count(post_processed_text: Text):
    return post_processed_text.count(" ") + 1


def get_word_count_vncorenlp(text: Text):
    return len(tokenize(text)) + word_count_offset


def merge_title_and_content(title: Text, content: Text):
    title = title.strip()
    if title[-1] == ".":
        ret = title + " " + content
    else:
        ret = title + ". " + content
    return ret


def main():
    cms_file = "/home/data/Data/20220425/cms_content_processed.csv"
    df = pd.read_csv(cms_file)
    r, c = df.shape

    for it in range(r):
        content = df.iloc[it]["body_string"]
        e = embedding(content)
        print(e)
        break


if __name__ == '__main__':
    main()
