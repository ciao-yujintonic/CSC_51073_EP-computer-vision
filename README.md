# Hand-Tracked Radial Chord Instrument

> A computer-vision-based musical interface controlled through real-time hand and fingertip tracking.

## Project Information

- **Course:** Image Analysis & Computer Vision
- **Institution:** École Polytechnique
- **Professor:** Mathieu Bredif
- **Team Members:** Yujin Hong, Junsik Bang
- **Project Type:** Course Project

## Overview

The **Hand-Tracked Radial Chord Instrument** is an interactive musical system that allows users to select and play chords using hand gestures captured by a camera.

Two radial menus are displayed over the live camera feed:

- The **left menu** controls the chord root note (A–G).
- The **right menu** controls the chord type (major, minor, suspended, etc.).

The project focuses on computer vision and human–computer interaction, particularly fingertip tracking, pinch detection, gesture timing, spatial selection, and robust interaction design.

## Initial Idea

The original concept was to recognize predefined hand gestures and map them to musical chords. Several interaction methods were considered:

- Using both hands to control chords
- Using one hand for chord selection and the other for octave control
- Using head movements to trigger drum sounds

## Professor's Feedback

The main feedback was to strengthen the **computer vision component** instead of focusing primarily on sound generation.

Additional suggestions included:

- Avoid combining hand and head tracking if it makes the interaction overly complex.
- Explore looping either chords or drums while performing the other part live.
- Compare different gesture-based selection techniques.

## Revised Concept

Based on the feedback, the project was refined into a hand-tracked radial chord instrument.

### Radial Menu Layout

The live camera interface contains two pie-shaped selection menus:

| Menu | Hand | Function | Interaction |
| --- | --- | --- | --- |
| Left radial menu | Left hand | Select a root note (A–G) | Hover over a segment |
| Right radial menu | Right hand | Select a chord type | Pinch within a segment |

### Left Hand: Root Note Selection

The left hand selects the root note by moving a tracked fingertip over one of the segments in the left radial menu.

A neutral **safeguard area** is placed at the center of the menu. This helps prevent unwanted intermediate notes when the user moves between distant choices—for example, moving directly from A to D without accidentally triggering B and C.

### Right Hand: Chord Type and Rhythm

The right hand selects a chord type by pinching inside a segment of the right radial menu.

The pinch gesture also provides temporal musical control:

- **Pinch frequency** determines the beat rate.
- **Pinch-hold duration** determines how long the current chord is played.

## Interaction Methods to Compare

The project will compare two selection techniques:

### Dwell-Based Selection

The user selects an option by keeping the fingertip inside a menu segment for a predefined amount of time.

### Pinch-Based Selection

The user points at a menu segment and confirms the selection by pinching.

The comparison may consider:

- Selection speed
- Selection accuracy
- Accidental activation rate
- Ease of use
- User fatigue

## Computer Vision Focus

The main computer vision tasks are:

- Real-time hand detection and tracking
- Fingertip localization
- Left- and right-hand identification
- Pinch detection based on finger positions
- Hover and dwell-state recognition
- Mapping image coordinates to radial-menu segments
- Temporal analysis of pinch frequency and hold duration
- Stabilization against tracking noise and unintended selections

## Planned Musical Features

- Root-note and chord-type selection
- Beat-rate control through repeated pinches
- Chord-duration control through pinch holding
- Chord progression recording and looping
- Potential live drum performance over a chord loop

## Current Interaction Flow

1. The camera detects and tracks both hands.
2. The left fingertip hovers over the left radial menu to select a root note.
3. The right hand points to a chord-type segment and confirms it with a pinch.
4. Pinch timing controls the beat rate and chord duration.
5. Selected chords may be recorded into a progression loop.
6. The user may perform an additional musical layer, such as drums, over the loop.

## Idea Sketch Update — November 4

The latest design separates the responsibilities of the two hands:

- **Left hand:** root-note selection by hovering, with a central safeguard zone
- **Right hand:** chord-type selection by pinching
- **Repeated pinches:** beat-rate control
- **Pinch-hold time:** current chord-duration control

This design keeps the interaction focused on hand tracking while providing enough visual and temporal information for a meaningful computer vision study.

## Team

- **Yujin Hong**
- **Junsik Bang**

Developed for the **Image Analysis & Computer Vision** course at **École Polytechnique**.
