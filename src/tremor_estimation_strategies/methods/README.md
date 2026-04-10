# Tremor Estimation algorithms reference papers
This is a list of the most referenced works on the field of tremor estimation/removal of human limb movement,
and vary from straightforward linear filters to deep-learning based signal analyses.
Some papers may introduce multiple methods, or have only a secondary focus on tremor estimation, with tremor suppression being the primary objective.

## Benedict-Bordner Filter (BBF) [Green for testing with controller!]
Paper name in methods/papers:
- 1962 - Synthesis of an Optimal Set of Radar Track-While-Scan Smoothing Equations

See also:
- 1998 - Tracking and Kalman filtering made easy
- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data

## Critically Damped Filter (CDF) [Green for testing with controller!]
Paper name in methods/papers:
- 1998 - Tracking and Kalman filtering made easy

See also:
- 1998 - Tracking and Kalman filtering made easy
- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data

## Fourier Linear Combiner (FLC) [Green for testing with controller!]
Paper name in methods/papers: 
- 1989 - Adaptive Fourier Estimation of Time-Varying Evoked Potentials 

## Weighted-frequency Fourier Linear Combiner (WFLC) [Green for testing with controller!]
Paper name in methods/papers: 
- 1995 - Adaptive human-machine interface for persons with tremor

See also:
- 1996 - Modeling and Canceling Tremor in Human-Machine Interfaces
- 1997 - Adaptive Fourier modeling for quantification of tremor
- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data

## Optimal Digital Filtering (ODF) [Won't implement]
Paper name in methods/papers:
- 2000 - Optimal digital filtering for tremor suppression

## Bandlimited Multiple Fourier Linear Combiner (BMFLC) [Green for testing with controller!]
Paper name in methods/papers:
- 2007 - Bandlimited Multiple Fourier Linear Combiner for Real-time Tremor Compensation

See also:
- 2010 - Estimation and filtering of physiological tremor for real-time compensation in surgical robotics applications
- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data

## Adaptive band-pass filter (ABPF) [Green for testing with controller!]
Paper name in methods/papers:
- 2010 - Adaptive band-pass filter (ABPF) for tremor extraction from inertial sensor data

## Weighted-frequency Fourier Linear Combiner with Kalman Filter (WFLC-KF) [Green for testing with controller!]
Paper name in methods/papers:
- 2010 - Real-time estimation of tremor parameters from gyroscope data

See also:
- 2013 - A neuroprosthesis for tremor management through the control of muscle co-contraction
- Obs.: this method also uses a critically damped g-h filter (CDF), to feed the WFLC.

## Bandlimited Multiple Fourier Linear Combiner with Recursive Least Squares (BMFLC-RLS) [Green for testing with controller!]
Paper name in methods/papers:
- 2011 - Estimation of Physiological Tremor from Accelerometers for Real-Time Applications

## Bandlimited Multiple Fourier Linear Combiner with Kalman Filter (BMFLC-KF) [Reasonable results if P isn't updated a posteriori]
Paper name in methods/papers:
- 2011 - Estimation of Physiological Tremor from Accelerometers for Real-Time Applications

## - HHTF - Hilbert–Huang-based Filtering [Won't implement: complicated, and had struggles with online prediction]
Paper name in methods/papers:
- 2011 - Hilbert–Huang-Based Tremor Removal to Assess Postural Properties From Accelerometers

## Autoregressive Least Mean Squares (AR-LMS) [Poor results with current implementation]
Paper name in methods/papers:
- 2013 - Physiological Tremor Estimation with Autoregressive (AR) Model and Kalman Filter for Robotics Applications

## Autoregressive Kalman Filter (AR-KF) [Poor results with current implementation]
Paper name in methods/papers:
- 2013 - Physiological Tremor Estimation with Autoregressive (AR) Model and Kalman Filter for Robotics Applications

## Multistep Weighted-frequency Fourier Linear Combiner with Kalman Filter (MS-WFLC-KF) [Won't implement: multistep predictions aren't useful] 
Paper name in methods/papers:
- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications

## Multistep Bandlimited Fourier Linear Combiner with Least Mean Squares (MS-BMFLC-LMS) [Won't implement: multistep predictions aren't useful]
Paper name in methods/papers:
- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications

## Multistep Bandlimited Fourier Linear Combiner with Kalman Filter (MS-BMFLC-KF) [Won't implement: multistep predictions aren't useful]
Paper name in methods/papers:
- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications

## Multistep Autoregressive Least Mean Squares (MS-AR-LMS) [Won't implement: multistep predictions aren't useful]
Paper name in methods/papers:
- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications

## Multistep Autoregressive Kalman Filter (MS-AR-KF) [Won't implement: multistep predictions aren't useful]
Paper name in methods/papers:
- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications

## High-pass Filter [Green for testing with controller!]
Paper name in methods/papers:
- 2014 - Robust Controller for Tremor Suppression at Musculoskeletal Level in Human Wrist 
- Obs.: This paper is not focused on tremor estimation, but it uses a simple bandpass filter to extract the tremor component from the signal, and it is one of the most cited papers in the field of tremor estimation. The bandpass filter is designed in the Laplace domain (eq. 1) with 4 zeroes and 4 poles. Its 8 coefficients are fixed.

## Adaptive Sliding Bandlimited Multiple Fourier Linear Combiner (ASBMFLC) [Green for testing with controller!]
Paper name in methods/papers:
- 2014 - Adaptive sliding bandlimited multiple fourier linear combiner for estimation of pathological tremor

## Least Squares Support Vector Machine (LS-SVM) [Won't implement: introduces delays]
Paper name in methods/papers:
- 2015 - Multistep prediction of physiological tremor based on machine learning for robotics assisted microsurgery

## Moving Window Least Squares Support Vector Machine (MWLS-SVM) [Won't implement: introduces delays]
Paper name in methods/papers:
- 2015 - Multistep prediction of physiological tremor based on machine learning for robotics assisted microsurgery

## Enhanced Bandlimited Multiple Fourier Linear Combiner (EBMFLC) [Green for testing with controller!]
Paper name in methods/papers:
- 2016 - Characterization of Upper-Limb Pathological Tremors: Application to Design of an Augmented Haptic Rehabilitation System

## Zero-Phase Adaptive Fuzzy Kalman Filter (ZPAFKF) [Poor results with current implementation]
Paper name in methods/papers:
- 2016 - A zero phase adaptive fuzzy Kalman filter for physiological tremor suppression in robotically assisted minimally invasive surgery

## Adaptive Multiple Oscillators Linear Combiner (AMOLC) [Poor results, pluts voluntary motion estimation uses BMFLC-KF (which also had issues)]
Paper name in methods/papers:
- 2016 - Prediction of pathological tremor using adaptive multiple oscillators linear combiner

## Enhanced High-Order WFLC-based Kalman Filter (EHWFLC-KF) [Poor results with current implementation]
Paper name in methods/papers:
- 2018 - Characterization of parkinsonian hand tremor and validation of a high-order tremor estimator.pdf

## Zero-Phase High-Pass Filter (ZPHP) [Won't implement: parameters are tied to Repetitive Control controller]
Paper name in methods/papers:
- 2019 - Repetitive Control of Electrical Stimulation for Tremor Suppression
- Obs.: although this paper's focus is tremor suppression through functional electrical stimulation, it proposes a simple zero-phase high-pass filter in the z plane to isolate tremor. Properties of the filter are discussed in detail, as well as its effects on control.

## Wavelet decomposition coupled with adaptive Kalman filtering for pathological tremor extraction (WAKE) [Won't implement: complicated, and not much better than EBMFLC]
Paper name in methods/papers:
- 2019 - WAKE: Wavelet decomposition coupled with adaptive Kalman filtering for pathological tremor extraction

## Adaptive Frequency Estimator (AFE) [Won't implement: estimates frequency, not amplitude]
Paper name in methods/papers:
- 2019 - Adaptive notch filter for pathological tremor suppression using permanent magnet linear motor
- Obs.: this works tremor suppression strategy is based on tremor frequency estimation, and not the tremor signal itself. Input signal is first filtered by a 6th order high-pass Butterworth filter, and the output of this operation is fed to a 2nd-order bandpass filter centered at the last estimated tremor frequency; only then is AFE used to estimate the new tremor frequency. According to equation 13, AFE uses both the high-pass output as well as the band-pass output to estimate tremor frequency.

## Extended Kalman Filter (EKF) [Won't implement: requires torque characterization]
Paper name in methods/papers:
- 2019 - Tremor Estimation and Removal in Robot-Assisted Surgery Using Lie Groups and EKF

## Quaternion Broad Learning System (QBLS) [Won't implement: struggles with real time requirements]
Paper name in methods/papers:
- 2020 - Quaternion broad learning system: A novel multi-dimensional filter for estimation and elimination tremor in teleoperation

## Three-Domain Fuzzy Wavelet Broad Learning System (TDFW-BLS) [Won't implement: struggles with real time requirements]
Paper name in methods/papers:
- 2020 - Three-domain fuzzy wavelet broad learning system for tremor estimation

## One-Dimensional Convolutional-Multilayer Perceptron (1D-CNN-MLP) [Won't implement: groundtruth for voluntary is a (zero-phase) butterworth lowpass filter]
Paper name in methods/papers:
- 2021 - Real-Time Voluntary Motion Prediction and Parkinson's Tremor Reduction Using Deep Neural Networks

## Long Short-Term Memory (LSTM) [Won't implement: requires offline training, and struggles with real time requirements for large prediction horizons]
Paper name in methods/papers:
- 2022 - Prediction of Pathological Tremor Signals Using Long Short-Term Memory Neural Networks