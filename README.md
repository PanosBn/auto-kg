
<p align="center">
  <img src="static/tue.png" alt="Logo TUe" width="30%" hspace="40" vspace="30"/>
  <img src="static/kpn.webp" alt="Logo KPN" width="30%"/>

</p>

# Triplet Extraction Framework for Knowledge Graph Population

This repository contains the implementation details and code for my thesis project, which focuses on developing a Triplet Extraction framework to populate an industrial Knowledge Graph, particularly for identifying customer problems in dialogues at KPN.

## Abstract

With the advent of deep learning techniques, state-of-the-art results have been achieved in many NLP tasks, including Entity Recognition and Relation Classification. These techniques, however, require a large amount of training data to be effective. This work proposes a pipeline for creating a Triplet Extraction framework, aiming to populate an industrial Knowledge Graph while reducing the resources required for building such a system. The uniqueness of this work lies in addressing Triplet Extraction for dialogues in customer-agent interactions at KPN. The project is divided into four main subtasks:
1. Creating the modelling and schema requirements of the Knowledge Graph.
2. Building a weak supervision framework for programmatically generating training data.
3. Training deep learning models for Entity Recognition and Relation Classification.
4. Integrating these techniques to generate a usable Knowledge Graph.

We demonstrate that weak supervision can, to some extent, replace human labeling and effectively train Entity Recognition models with weakly supervised data.

## Repository Structure

- `Preprocessing`: Contains scripts used for creating the training data.
- `weak_supervision`: Methods used for creating the weak supervision framework.
- `models`: Code for training the Entity Recognition and Relation Classification models.

---

This repository serves as a comprehensive guide and practical implementation of the techniques and methodologies discussed in the thesis.

---
