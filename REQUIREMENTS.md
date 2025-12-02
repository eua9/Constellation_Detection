Project: Constellation Detection via Classical Computer Vision
Part 1: Context & Analysis
1. Analysis of the Current Project Scope
Goal: Automatically detect constellations in astronomical images by identifying star patterns and matching them to known templates. The system must be robust to scale, rotation, noise, and partial visibility.
Data: Public star field images (NASA, ESA, Kaggle) plus optional synthetic star maps based on catalogs like Hipparcos and the Bright Star Catalog.
Method (Pipeline):
Star detection: Using thresholding or blob detection to extract star centroids.
Graph construction: Computing pairwise distances and normalizing patterns using PCA for rotation/scale invariance.
Constellation identification: Comparing star graphs to templates using Procrustes analysis, Hausdorff distance, or general graph matching.
Strengths:
Ambitious, clearly motivated real-world task (Astronomy + Computer Vision).
Uses a clean classical CV pipeline: detection → normalization → matching.
Already mentions PCA normalization and graph-based representation, which are solid foundations.
Acknowledges the need for synthetic star maps for data augmentation and debugging.
Gaps / Risks vs. Feedback:
Matching step complexity: Jumping directly to Procrustes/Hausdorff is too heavy for a starting point.
Missing baseline: No explicit simple baseline (e.g., PCA-normalized coordinates + Sum of Squared Differences).
Synthetic data specification: The plan for synthetic data is currently under-specified (needs staged generation of rotated/scaled/noisy variants).
Star detection robustness: Needs thorough validation before moving to matching.
Evaluation: Lacks detailed metrics (accuracy, precision/recall) and explicit experiment plans.
2. Analysis of Professor’s Requested Changes
The feedback requires a shift in approach rather than a change in the problem statement.
Start with a simple graph matching baseline:
After rotation/scale normalization (via PCA), compute a simple distance (e.g., Sum of Squared Differences - SSD) between star coordinates.
This becomes Step 3a (Baseline) before implementing advanced methods.
Build up complexity gradually:
Pipeline: Baseline $\rightarrow$ Improved Methods $\rightarrow$ (Optional) Advanced Graph Matching.
Make synthetic data central:
Explicitly create synthetic "query" images by rotating, scaling, and adding noise to known templates.
Use this to measure how often the algorithm correctly identifies the constellation against ground truth.
Prioritize robust star detection:
Verify star centroids are accurate and stable using tools like scikit-image.

Part 2: Formal Project Proposal
3. Revised Project Scope
Project Title: Constellation Detection via Classical Computer Vision
Course: CS 4337 – Intro to Computer Vision
Student: Ryan Mitchell
1. Problem Statement
The goal of this project is to automatically detect and identify constellations in astronomical images using a classical computer vision pipeline. Given an input star field (real or synthetic), the system should recognize which known constellation is present, even under rotation, scale changes, noise, missing stars, extra stars, and partial visibility.
2. High-Level Approach
The project will implement a three-stage pipeline:
A. Star Detection
Detect star-like blobs in grayscale astronomical images.
Extract star centroids ($x, y$ coordinates in image space).
B. Star Graph Construction & Normalization
Represent the set of detected stars as a point set/graph.
Normalize for translation, rotation, and scale using PCA and centering so that constellations with the same structure align in a canonical frame.
C. Constellation Matching (Staged Complexity)
Baseline (Required): After normalization, match the query point set to each template by computing a simple Sum of Squared Differences (SSD) between star coordinates and choosing the template with minimal SSD.
Improved Methods (Stretch Goals):
Hausdorff distance between point sets.
RANSAC-based alignment to handle outliers/extra stars.
Advanced graph matching strategies.
The focus will be on building a simple, working baseline first, then incrementally adding more sophisticated matching techniques.
3. Dataset & Synthetic Data
Real Images: Publicly available star field images (NASA, ESA, Kaggle).
Synthetic Star Maps (Core): A generator that starts from known templates and applies:
Rotation, scale, and translation transformations.
Gaussian noise on star positions.
Spurious stars (clutter) or removal of stars (occlusion).
Ground truth tracking for quantitative evaluation.
4. Tools & Libraries
Python: Core language.
NumPy/SciPy: Numerical operations.
OpenCV: Basic image processing.
scikit-image: Blob detection (blob_log, find_contours) and geometric transforms.
Matplotlib: Visualization and debugging.
5. Planned Deliverables
Python Implementation:
Star detection module.
Normalization and graph construction module.
SSD-based matching baseline.
Synthetic dataset generator.
Experimental Results:
Quantitative evaluation on synthetic data.
Qualitative examples on real astronomy images.
Project Report: Methods, implementation details, results, and limitations.

Part 3: Project Requirements Specification
4. Functional Requirements
ID
Requirement
Description
FR1
Input Handling


FR1.1
Image Input
The system shall accept grayscale astronomical images (real or synthetic).
FR1.2
Batch Processing
The system shall support batch evaluation on a directory of test images.
FR2
Star Detection


FR2.1
Blob Detection
The system shall detect star-like blobs in the input image.
FR2.2
Centroid Output
The system shall output a list of star centroids as 2D coordinates.
FR2.3
Configurability
The detection method shall be configurable (threshold, blob size).
FR2.4
Visualization
The system shall provide a visualization mode overlaying detected stars for debugging.
FR3
Graph Construction


FR3.1
Point Set
The system shall represent detected stars as a point set in image coordinates.
FR3.2
Normalization
The system shall normalize the point set for translation (centering), rotation (PCA), and scale.
FR4
Template Management


FR4.1
Template Storage
The system shall store known constellation templates as normalized point sets.
FR4.2
Management
The system shall allow adding/removing templates via configuration.
FR5
Baseline Matching
(Required)
FR5.1
Pipeline Execution
For each query, the system shall detect stars, normalize coordinates, and compare against templates.
FR5.2
SSD Algorithm
The baseline shall compute Sum of Squared Differences (SSD) between the normalized query and templates.
FR5.3
Output
The system shall output the best-matching constellation name and the SSD score.
FR6
Extended Matching
(Optional / Stretch)
FR6.1
Hausdorff Distance
The system may implement Hausdorff distance comparison between point sets.
FR6.2
RANSAC
The system may implement RANSAC-based matching for robustness against outliers.
FR7
Synthetic Generator


FR7.1
Generation
The system shall generate synthetic images based on templates with configurable transformations (rotation, scale, noise).
FR7.2
Ground Truth
The generator shall save the rendered image and the ground truth constellation label.
FR8
Evaluation


FR8.1
Metrics
The evaluation script shall compute classification accuracy on the synthetic dataset.
FR8.2
Confusion Matrix
The script shall report a confusion matrix over constellations.

5. Non-Functional Requirements & Milestones
Non-Functional Requirements
NFR1 Performance: Processing a $512 \times 512$ image should take less than 5 seconds on a standard laptop.
NFR2 Code Structure: Code must be modular (star_detection.py, normalization.py, etc.) with clear docstrings.
NFR3 Reproducibility: Dependencies must be listed in requirements.txt. Synthetic generation must accept a random seed.
Suggested Milestones
Milestone 1: Implement/debug star detection on synthetic images and visualize results.
Milestone 2: Implement PCA-based normalization; verify alignment of rotated/scaled constellations.
Milestone 3: Implement SSD baseline matching and run initial evaluation.
Milestone 4: (Stretch) Add advanced matching (Hausdorff/RANSAC) and compare performance.
