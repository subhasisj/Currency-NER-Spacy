import re

class TextPreprocessor:
    
    def __init__(self,logger):
        self.logger = logger

    def _remove_urls(self,text):
        return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

    def _remove_numeric(self,text):
        return re.sub(r'\d+', '', text) 

    def _remove_extra_spaces(self,text):

        lines = text.split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

        return (' '.join(non_empty_lines))

    def _remove_text_between_brackets(self,text):
        return re.sub("[\(\[].*?[\)\]]", "", text)

    def clean_text(self,text): 

        try:
            self.logger.log(f'Cleaning: removing extra spaces..')
            text = self._remove_extra_spaces(text)
            # self.logger.log(f'Cleaning: removing numeric values..')
            # text = self._remove_numeric(text)
            self.logger.log(f'Cleaning: removing URL\'s..')
            text = self._remove_urls(text)
            self.logger.log(f'Cleaning: removing text between brackets..')
            text = self._remove_text_between_brackets(text)

            return text

        except Exception as e:
            self.logger.log(f'Cleaning: Exception occured, Exception : {str(e)}','critical')
            raise Exception




