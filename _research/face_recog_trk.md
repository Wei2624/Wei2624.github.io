---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: A Super-Fast Online Face Tracking System for Video Surveillance
share: false
permalink: /research/face_recog_trk/
---

This work was performed at Microsoft Research Asia where I was advised by Dr. Baining Guo and Dr. Wenjun Zeng for the full-time research internship. 

# Overview

As the number of surveillance cameras has been dramatically growning in the past decade, video understanding and analysis have become an important and popular research topic in computer vision. In particular, online face tracking has attracted much attention to facilitate a wide range of applications. 

In this paper, we propose a novel and practical system for robust online face tracking in surveillance videos. In general, the online face tracking system has two components: face detector and face tracker. Initially, the system launches the face detector and triggers the face tracker. Then, the face tracker runs on each successive frame until a predefined frame number $N_1$ is reached or a tracker failure is recognized. In the failure case, the information of the failed tracker is recorded in a buffer. The face detector is then triggered again to provide anchor bounding boxes. If  there  are  faces  detected,  the  updating  module  is  triggered and  the  active  tracklets  are  updated  based  on  the  detection results as well as the buffered tracklets. Otherwise, we check whether there are still active trackles. The figure below shows the framework of our proposed online face tracking system. 

![Framework](/_research/images/face_recog_trk_1.png)

# Demo

A real demo with explanation can be viewed below. 

![Framework](/_research/images/face_recog_trk_2.png)

Details can be found in our paper. 

# My Contributions

In this work, I mainly focus on two parts. The first part is face detector. I was responsible for designing an accurate face detector and improving it based on the video surveillance assumption. The second part is merging face tracker with face detector. The resulted pipeline was exported into a public API hosted in Azure. 