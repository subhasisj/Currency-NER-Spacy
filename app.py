import streamlit as st
import spacy
import os
from textblob import TextBlob
from src.text_preprocessor import TextPreprocessor
from src.logger import Logger
from datetime import datetime

def main():
    time_created = datetime.now()
    logger = Logger(f'inference_logs_{time_created.date()}_{time_created.strftime("%H%M%S")}.log')

    st.header('Currency Entity Recognition')
    # st.image('./images/thyroid_1.png',clamp=True,width=500)
    user_input = st.text_area("Text to analyze")
    text_preprocessor = TextPreprocessor(logger)
    if st.button('Analyze'):

        try:
            model = load_model()
            st.info("Model Loaded successfully")
        except Exception as e:
            st.warning(f"Unable to load model, {str(e)}")

        # Check sentiment, only in case an opinion or a sentiment is expressed, then we extract entities
        processed_text = TextBlob(user_input)

        # Check sentence wise sentiment and extract entities
        entities_found = []
        for s in processed_text.sentences:
            if s.sentiment.polarity > 0.05 or s.sentiment.polarity < -0.05:
                text = text_preprocessor.clean_text(s)

                doc = model(text)

                for ent in doc.ents:

                    entities_found.append(ent.label_)
                # st.write('Entities',[(ent.label_) for ent in doc.ents if ent.text is not ''])
        if len(entities_found) > 0:  
            st.success(f'Unique Currency Entities Found: {len(set(entities_found))}')
            st.write(set(entities_found))
        else:
            st.warning('No entities found.')




@st.cache(allow_output_mutation=True)
def load_model():
    return spacy.load(os.path.join('.','model'))


if __name__ == '__main__':
    main()