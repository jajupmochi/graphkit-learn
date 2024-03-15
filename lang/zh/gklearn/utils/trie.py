#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:48:49 2019

Trie (prefix tree)
@author: ljia
@references:
`NLP: Build a Trie Data structure from scratch with python <https://viblo.asia/p/nlp-build-a-trie-data-structure-from-scratch-with-python-3P0lPzroKox>`__, 2019.1
"""

import pickle
import json

""" Trie class
"""
class Trie:
    """
    """
    # init Trie class
    def __init__(self):
        self.root = self.getNode()

    def getNode(self):
        return {"isEndOfWord": False, "children": {}}

    def insertWord(self, word):
        current = self.root
        for ch in word:

            if ch in current["children"]:
                node = current["children"][ch]
            else:
                node = self.getNode()
                current["children"][ch] = node

            current = node
        current["isEndOfWord"] = True
        if 'count' in current:
            current['count'] += 1
        else:
            current['count'] = 1

    def searchWord(self, word):
        current = self.root
        for ch in word:
            if ch not in current["children"]:
                return 0
            node = current["children"][ch]

            current = node
        if 'count' in current:
            return current["count"]
        else:
            return 0

    def searchWordPrefix(self, word):
        current = self.root
        for ch in word:
            if not current["children"].has_key(ch):
                return False
            node = current["children"][ch]

            current = node
        # return True if children contain keys and values
        return bool(current["children"])

    def deleteWord(self, word):
        self._delete(self.root, word, 0)

    def _delete(self, current, word, index):
        if(index == len(word)):
            if not current["isEndOfWord"]:
                return False
            current["isEndOfWord"] = False
            return len(current["children"].keys()) == 0

        ch = word[index]
        if not current["children"].has_key(ch):
            return False
        node = current["children"][ch]

        should_delete_current_node = self._delete(node, word, index + 1)

        if should_delete_current_node:
            current["children"].pop(ch)
            return len(current["children"].keys()) == 0

        return False

    def save_to_pickle(self, file_name):
        f = open(file_name + ".pkl", "wb")
        pickle.dump(self.root, f)
        f.close()

    def load_from_pickle(self, file_name):
        f = open(file_name + ".pkl", "rb")
        self.root = pickle.load(f)
        f.close()
        
    def to_json(self):
        return json.dump(self.root)

    def save_to_json(self, file_name):
        json_data = json.dumps(self.root)
        f = open(file_name + ".json", "w")
        f.write(json_data)
        f.close()

    def load_from_json(self, file_name):
        json_file = open(file_name + ".json", "r")
        self.root = json.load(json_file)
        json_file.close()
