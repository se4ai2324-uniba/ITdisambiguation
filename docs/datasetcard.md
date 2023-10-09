## Dataset Summary

The SemEval-2023 Visual Word Sense Disambiguation dataset is composed of 12896 entries in the training set and 463 in the test set.
Given an ambiguous word and some limited textual context, the task is to select among a set of candidate images the one which corresponds to the intended meaning of the target word.

Every sample has the following features:
- **Target word:** an ambiguous word associated to the target image
- **Context:** another word giving the context of the target word
- **Images:** a list of ten ambiguous images that could be associated to the target word
- **Target image:** the correct image among the ten

<p align="center">
  <img src="dataset.png" width="500px"/>
</p>

## Using the Dataset
This dataset can be used to train a model to associate an image to a word that has more than one intended meaning when we have only a limited context.

## More informations
More informations about the dataset can be found on the [official SemEval-2023 website](https://raganato.github.io/vwsd/).
