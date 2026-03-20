import re
from typing import Dict


class TextNormalizer:
    """文本规范化策略"""
    
    DEFAULT_REPLACEMENTS = {
        "\\[": "$$", "\\]": "$$", "\\(": "$", "\\)": "$",
        '\u2018': "'", '\u2019': "'", '\u0060': "'", 
        '\u201C': '"', '\u201D': '"', 
        '\u2013': '-', '\u2014': '--', '\u2212': '-', 
        '\u00A0': ' ', '\u2026': '...'
    }
    
    def __init__(self, custom_rules: Dict[str, str] = None):
        self.replacements = {**self.DEFAULT_REPLACEMENTS, **(custom_rules or {})}
        self._pattern = re.compile('|'.join(re.escape(k) for k in self.replacements))
    
    def normalize(self, text: str) -> str:
        """规范化文本"""
        return self._pattern.sub(lambda m: self.replacements[m.group()], text)


def comprehensive_normalization(text: str) -> str:
    """向后兼容的函数接口"""
    return TextNormalizer().normalize(text)
