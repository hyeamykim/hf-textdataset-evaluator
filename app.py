import streamlit as st

def main():

    st.set_page_config(
        page_title="HuggingFace Text Dataset Evaluator",
        page_icon=":white_check_mark:",
    )

    st.write('''
             # :white_check_mark: HuggingFace Text Dataset Evaluator
             '''
             )

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        The quality of the text datasets is increasingly important for any NLP downstream applications.
        This app helps you check for **Gender Bias, Toxicity, and Personally Identifiable Informationn (PII)**
        in any HuggingFace text dataset.

        **👈 Check out the demos from the sidebar** to see some examples!

        - **Gender Bias Demo**: Check if the text dataset contains imbalanced gender representations with gender magnitude metrics.

        - **Toxicity Demo**: Check if the text dataset contains toxic language with tosicitcy classification models. 

        - **PII Demo**: Check if the text dataset contains PII with Microsoft Presidio API.        
        """
    )
    

if __name__ == '__main__':
    main()

