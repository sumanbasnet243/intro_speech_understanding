import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.

    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio

    @returns:
    text (str) - the recognized speech
    '''

    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)   # Read the entire audio file

    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Speech could not be understood."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"