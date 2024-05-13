# Classification of Vibration Medium & Predicting the Likelihood of Damage

- The problem being addressed here is a multilabel classification with the help of a neural network. A tri-axial accelerometer and a gyroscope were used to measure the vibration and angular velocity generated by two different mediums (shakers).
- Based on the signals generated, a neural network was designed to classify the signals into different mediums and predict the likelihood of medium damage.

- `[Stationary or Ground Zero dataset](Project/stationary_output.csv)`: This dataset was used as ground zero to correct the offset in measured signals.
- `[Medium 1](Project/vib_30hz_upright.csv)`: This dataset represents `Medium_1` in the code.
- `[Medium 2](Project/vib_30hz_rig.csv)`: This dataset represents `Medium_2` in the code.
- Each of these shakers operate at 30Hz but are input with a different sine wave causing them to vibrate at different frequencies. These vibrations are measured using the IMU and the neural network is used for the classification tasks. 
