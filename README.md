# Music Generation with MIDI and LSTM

This project demonstrates how to generate music using MIDI files and an LSTM (Long Short-Term Memory) model in TensorFlow. The goal is to preprocess MIDI data, train an LSTM model, and generate new musical compositions.

## Requirements

Before running the code, make sure to install the necessary dependencies:

```bash
sudo apt install -y fluidsynth
pip install --upgrade pyfluidsynth
pip install pretty_midi
pip install tensorflow
pip install seaborn
```

## Project Structure

- **Data**: The dataset used in this project is the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro), which contains piano performances in MIDI format.
- **Model**: The project uses an LSTM model for sequence prediction, trained to predict the next note in a musical sequence.
- **Functions**:
  - `midi_to_notes`: Converts MIDI files to a dataframe of notes (pitch, step, duration).
  - `plot_piano_roll`: Plots a piano roll representation of the notes in a given MIDI file.
  - `notes_to_midi`: Converts a sequence of notes back into a MIDI file.
  - `create_sequences`: Prepares the data by creating input-output pairs for training the LSTM model.
  - `predict_next_note`: Uses the trained model to predict the next note in a sequence.

## How It Works

1. **Data Loading**:
   - MIDI files from the MAESTRO dataset are loaded and converted into a structured format (notes, step, duration).
   
2. **Data Preprocessing**:
   - The MIDI data is converted into a sequence of notes using the `midi_to_notes` function.
   - The sequences are then prepared using `create_sequences`, and split into training data.

3. **Model**:
   - An LSTM model is trained to predict the next note in a musical sequence. The model has three output layers:
     - `pitch` (categorical)
     - `step` (regression)
     - `duration` (regression)
   
4. **Music Generation**:
   - The trained model is used to predict new notes, which are then converted back into a MIDI file using `notes_to_midi`.
   
5. **Audio Playback**:
   - The generated MIDI file can be played using `fluidsynth` for audio output.

## Example Usage

```python
import pretty_midi
from IPython import display

# Load a sample MIDI file
sample_file = 'path/to/your/midi/file.mid'
pm = pretty_midi.PrettyMIDI(sample_file)

# Display the first 30 seconds of audio
display.display_audio(pm, seconds=30)
```

To generate a new music sequence:

```python
# Generate new notes using the trained model
generated_notes = []
for _ in range(120):  # Number of notes to generate
    pitch, step, duration = predict_next_note(input_notes, model, temperature=1.0)
    generated_notes.append((pitch, step, duration, start_time, end_time))

# Convert the generated notes back into a MIDI file
out_pm = notes_to_midi(generated_notes, 'output.mid', instrument_name="Acoustic Grand Piano")
display.display_audio(out_pm)
```

## Model Training

The model is trained using the following setup:

- Loss function:
  - `pitch`: Sparse categorical cross-entropy
  - `step` and `duration`: Mean squared error with positive pressure on negative step values
- Optimizer: Adam
- Callbacks: Model checkpoint and early stopping
