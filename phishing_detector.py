import logging
import numpy as np
import pandas as pd
import re
import email
from dotenv import load_dotenv
from datetime import datetime
from email import policy
from email.parser import BytesParser
import os
import traceback
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    accuracy_score,
    log_loss,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional, Union, Tuple, Any
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from textblob import TextBlob
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import hashlib
import time
from transformers import pipeline

# Initialize console for rich output
console = Console()

# Configure logging with production settings
LOG_PATH = os.getenv("LOG_PATH", "./logs/phishing_cli.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)s - %(message)s"

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)

if os.getenv("LOG_CONSOLE", "True").lower() in ["true", "1", "yes"]:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info("Phishing CLI started")

# Thread-safe cache for feature extraction with LRU eviction
class ThreadSafeCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.access_times = {}
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Remove least recently used item
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            self.cache[key] = value
            self.access_times[key] = time.time()

# Load .env with production settings
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./model/phishing_model.joblib")
CONFIG_PATH = os.getenv("CONFIG_PATH", "./config/model_config.json")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))
DATASET_PATH = os.getenv("DATASET_PATH", "C:/Users/msi/Documents/phishing_email_dataset.csv")

@dataclass
class ModelConfig:
    """Configuration for the phishing detection model."""
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    test_size: float = 0.2
    feature_threshold: float = 0.5
    confidence_threshold: float = 0.5
    nltk_data: Dict[str, List[str]] = field(default_factory=lambda: {
        "required_resources": [
            "punkt",
            "stopwords",
            "averaged_perceptron_tagger",
            "maxent_treebank_pos_tagger"
        ]
    })

# Pre-compile regex patterns for better performance
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Initialize NLTK data once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Initialize stopwords once
STOPWORDS = set(stopwords.words('english'))

class EmailPreprocessor:
    """Handles email content preprocessing with optimized methods."""
    
    def __init__(self):
        self.html_parser = BeautifulSoup("", "html.parser")
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
    @lru_cache(maxsize=1000)
    def clean_html(self, html_content: str) -> str:
        """Remove HTML tags and clean content with caching."""
        if not html_content:
            return ""
        try:
            self.html_parser = BeautifulSoup(html_content, "html.parser")
            return self.html_parser.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {str(e)}")
            return html_content
    
    @lru_cache(maxsize=1000)
    def normalize_text(self, text: str) -> str:
        """Normalize text with caching."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).lower().strip()
    
    @lru_cache(maxsize=1000)
    def extract_links(self, html_content: str) -> List[str]:
        """Extract all links with caching."""
        if not html_content:
            return []
        try:
            self.html_parser = BeautifulSoup(html_content, "html.parser")
            return [a.get('href') for a in self.html_parser.find_all('a', href=True)]
        except Exception as e:
            logger.warning(f"Error extracting links: {str(e)}")
            return []
    
    @lru_cache(maxsize=1000)
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses with caching."""
        if not text:
            return []
        return self.email_pattern.findall(text)
    
    def process_email(self, email_text: str) -> Dict[str, Any]:
        """Process email content efficiently."""
        try:
            # Parse email
            msg = BytesParser(policy=policy.default).parsebytes(email_text.encode())
            
            # Extract content
            text_content = []
            html_content = []
            
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    text_content.append(part.get_content())
                elif content_type == 'text/html':
                    html_content.append(part.get_content())
            
            # Process content
            clean_text = self.normalize_text('\n'.join(text_content))
            clean_html = self.clean_html('\n'.join(html_content))
            links = self.extract_links(clean_html)
            emails = self.extract_emails(clean_text)
            
            return {
                'text': clean_text,
                'html': clean_html,
                'links': links,
                'emails': emails,
                'headers': dict(msg.items())
            }
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
            return {
                'text': '',
                'html': '',
                'links': [],
                'emails': [],
                'headers': {}
            }

class TextAnalyzer:
    """Optimized text analysis for phishing detection with focus on high-precision features."""
    
    def __init__(self):
        # Pre-compile only the most reliable regex patterns
        self.patterns = {
            'urgency': re.compile(r'urgent|immediately|verify|important|warning|suspend|action required|hurry|asap|limited time|expires|deadline|last chance|final notice|overdue|payment due'),
            'phishing': re.compile(r'account|password|login|verify|click here|dear customer|confirm|security|alert|update|validate|authenticate|credentials|bank|credit card|ssn|social security|invoice|payment|billing|finance|account payable'),
            'grammar': re.compile(r'\b(?:your|youre|ur|u|r)\b|\b(?:pls|plz|thx|tnx|ty)\b|\b(?:dear\s+valued\s+customer|dear\s+sir/madam)\b')
        }
        
        # Initialize NLP components with error handling
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.nlp_initialized = True
            logger.info("NLP components initialized successfully")
        except Exception as e:
            logger.warning(f"NLP initialization failed: {str(e)}")
            self.nlp_initialized = False
            self.sentiment_analyzer = None
    
    @lru_cache(maxsize=1000)
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text focusing on the most reliable phishing indicators."""
        if not text:
            return {}
            
        text = text.lower()
        results = {}
        
        # 1. Urgency and Phishing Pattern Analysis (Most Reliable)
        for category, pattern in self.patterns.items():
            matches = pattern.findall(text)
            # Normalize by text length and word count
            word_count = len(text.split())
            results[f'{category}_score'] = len(matches) / (word_count + 1) if word_count > 0 else 0.0
        
        # 2. NLP Analysis (When Available)
        if self.nlp_initialized:
            try:
                # Tokenize and analyze
                words = word_tokenize(text)
                if words:
                    # POS Tagging for Key Indicators
                    pos_tags = pos_tag(words)
                    
                    # Calculate key ratios
                    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
                    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
                    
                    # Focus on verb-to-noun ratio (strong phishing indicator)
                    results['verb_noun_ratio'] = verb_count / (noun_count + 1) if noun_count > 0 else 0.0
                    
                    # Sentiment Analysis (using TextBlob for reliability)
                    blob = TextBlob(text)
                    results['sentiment'] = blob.sentiment.polarity
                    
                    # Readability Analysis (important for phishing detection)
                    sentences = sent_tokenize(text)
                    if sentences:
                        avg_sentence_length = len(words) / len(sentences)
                        avg_word_length = sum(len(word) for word in words) / len(words)
                        
                        # Calculate Flesch Reading Ease Score (normalized)
                        readability_score = max(0.0, min(1.0, (206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length) / 100.0))
                        results['readability_score'] = readability_score
                        
                        # Calculate lexical diversity (important for spam detection)
                        unique_words = set(word for word in words if word not in STOPWORDS)
                        results['lexical_diversity'] = len(unique_words) / (len(words) + 1)
                    
            except Exception as e:
                logger.warning(f"Error in NLP analysis: {str(e)}")
                # Set default values for NLP features
                results.update({
                    'verb_noun_ratio': 0.0,
                    'sentiment': 0.0,
                    'readability_score': 0.5,
                    'lexical_diversity': 0.0
                })
        
        return results

class ContentAnalyzer:
    """Integrated content analysis for phishing detection."""
    
    def __init__(self):
        # URL analysis patterns
        self.url_patterns = {
            'suspicious_tlds': {'xyz', 'top', 'cc', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw'},
            'shorteners': {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'buff.ly'},
            'keywords': {'login', 'verify', 'account', 'password', 'secure', 'update', 'confirm'}
        }
        
        # Header analysis patterns
        self.header_patterns = {
            'suspicious_senders': [
                r'security.*alert', r'security.*team', r'support.*team',
                r'customer.*service', r'account.*services', r'security.*department'
            ]
        }
    
    def analyze(self, email_text: str) -> Dict[str, float]:
        """Integrated content analysis combining URL and header features."""
        results = {}
        
        # Extract URLs and headers
        links = self._extract_links(email_text)
        headers = self._extract_headers(email_text)
        
        # URL analysis
        results.update(self._analyze_urls(links))
        
        # Header analysis
        results.update(self._analyze_headers(headers))
        
        # HTML content analysis
        results.update(self._analyze_html(email_text))
        
        return results
    
    def _extract_links(self, text: str) -> List[str]:
        """Extract all links from text."""
        soup = BeautifulSoup(text, 'html.parser')
        return [a.get('href') for a in soup.find_all('a', href=True)]
    
    def _extract_headers(self, text: str) -> Dict[str, str]:
        """Extract email headers."""
        headers = {}
        for line in text.split('\n'):
            if line.strip() == '':
                break
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                key, value = match.groups()
                headers[key.lower()] = value.strip()
        return headers
    
    def _analyze_urls(self, urls: List[str]) -> Dict[str, float]:
        """Analyze URLs for suspicious patterns."""
        if not urls:
            return {
                'suspicious_url_count': 0.0,
                'url_entropy': 0.0,
                'shortened_url_count': 0.0,
                'suspicious_tld_count': 0.0
            }
        
        results = {
            'suspicious_url_count': 0.0,
            'url_entropy': self._calculate_entropy(urls),
            'shortened_url_count': 0.0,
            'suspicious_tld_count': 0.0
        }
        
        for url in urls:
            url = url.lower()
            
            # Check TLDs
            tld = url.split('.')[-1]
            if tld in self.url_patterns['suspicious_tlds']:
                results['suspicious_tld_count'] += 1
            
            # Check shorteners
            if any(shortener in url for shortener in self.url_patterns['shorteners']):
                results['shortened_url_count'] += 1
            
            # Check keywords
            if any(keyword in url for keyword in self.url_patterns['keywords']):
                results['suspicious_url_count'] += 1
        
        # Normalize counts
        total_urls = len(urls)
        for key in ['suspicious_url_count', 'shortened_url_count', 'suspicious_tld_count']:
            results[key] /= (total_urls + 1)
        
        return results
    
    def _analyze_headers(self, headers: Dict[str, str]) -> Dict[str, float]:
        """Analyze email headers."""
        results = {
            'header_inconsistency': 0.0,
            'reply_to_domain_match': 0.0,
            'spf_dkim_alignment': 0.0,
            'suspicious_sender': 0.0
        }
        
        if not headers:
            return results
        
        # Check header inconsistencies
        inconsistencies = 0
        if 'from' in headers and 'reply-to' in headers and headers['from'] != headers['reply-to']:
            inconsistencies += 1
        if 'return-path' in headers and 'from' in headers and headers['return-path'] != headers['from']:
            inconsistencies += 1
        if 'message-id' in headers and not re.match(r'<[^>]+@[^>]+>', headers['message-id']):
            inconsistencies += 1
        results['header_inconsistency'] = inconsistencies / 3.0
        
        # Check domain matching
        if 'from' in headers and 'reply-to' in headers:
            from_domain = re.search(r'@([^>]+)', headers['from'])
            reply_domain = re.search(r'@([^>]+)', headers['reply-to'])
            if from_domain and reply_domain and from_domain.group(1) == reply_domain.group(1):
                results['reply_to_domain_match'] = 1.0
        
        # Check SPF/DKIM
        auth_results = headers.get('authentication-results', '').lower()
        results['spf_dkim_alignment'] = 1.0 if 'spf=pass' in auth_results and 'dkim=pass' in auth_results else 0.0
        
        # Check suspicious sender
        if 'from' in headers:
            from_header = headers['from'].lower()
            matches = sum(1 for pattern in self.header_patterns['suspicious_senders'] 
                        if re.search(pattern, from_header))
            results['suspicious_sender'] = matches / (len(self.header_patterns['suspicious_senders']) + 1)
        
        return results
    
    def _analyze_html(self, text: str) -> Dict[str, float]:
        """Analyze HTML content."""
        return {
            'has_login_form': 1 if re.search(r'<form.*?>.*?</form>', text, re.DOTALL | re.IGNORECASE) else 0,
            'has_script': 1 if re.search(r'<script.*?>', text, re.IGNORECASE) else 0,
            'has_ip_url': 1 if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', text) else 0,
            'has_hex_ip': 1 if re.search(r'http[s]?://0x[0-9a-fA-F]+', text) else 0,
            'spoofed_display': 1 if re.search(r'From: .*<.*@gmail\.com>', text) and "company.com" in text else 0,
            'dmarc_fail': 1 if re.search(r'dmarc\s*=\s*fail', text.lower()) else 0,
            'spf_fail': 1 if re.search(r'spf\s*=\s*fail', text.lower()) else 0,
            'dkim_fail': 1 if re.search(r'dkim\s*=\s*fail', text.lower()) else 0,
            'has_attachment': 1 if re.search(r'Content-Disposition: attachment', text, re.IGNORECASE) else 0,
            'suspicious_ext': 1 if re.search(r'\.(exe|scr|js|zip|rar|xlsm|bat|jar|msi|vbs|ps1|cmd)', text, re.IGNORECASE) else 0,
            'double_ext': 1 if re.search(r'\.\w+\.(exe|scr|bat|js|vbs|ps1|cmd)', text, re.IGNORECASE) else 0,
            'encoded_chars': 1 if re.search(r'%[0-9a-fA-F]{{2}}|&#x|&#\d+;', text) else 0,
            'link_mismatch': 1 if re.search(r'<a\s+href="(.*?)">(.*?)</a>', text, re.IGNORECASE | re.DOTALL) and re.search(r'<a\s+href="(.*?)">(.*?)</a>', text, re.IGNORECASE | re.DOTALL).group(1) != re.search(r'<a\s+href="(.*?)">(.*?)</a>', text, re.IGNORECASE | re.DOTALL).group(2) else 0,
            'invisible_characters': 1 if re.search(r'\u200B|\u200C|\u200D|\u2060', text) else 0
        }
    
    def _calculate_entropy(self, urls: List[str]) -> float:
        """Calculate entropy of URL characters."""
        if not urls:
            return 0.0
        
        all_chars = ''.join(urls)
        char_freq = {}
        total_chars = len(all_chars)
        
        if total_chars == 0:
            return 0.0
        
        for char in all_chars:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        entropy = 0.0
        for freq in char_freq.values():
            prob = freq / total_chars
            entropy -= prob * np.log2(prob)
        
        max_entropy = np.log2(len(char_freq)) if char_freq else 0
        return entropy / max_entropy if max_entropy > 0 else 0.0

class PhishingDetector:
    def __init__(self, model_path: str = None):
        """Initialize phishing detector with optional model path."""
        self.model = self._load_model(model_path) if model_path else None
        
        # Feature names for display
        self.feature_names = {
            'suspicious_urls': "Suspicious URLs",
            'malicious_attachments': "Malicious Attachments",
            'spoofed_headers': "Spoofed Headers",
            'sensitive_keywords': "Sensitive Keywords",
            'security_indicators': "Security Indicators",
            'social_engineering': "Social Engineering",
            'security_grammar': "Security Grammar",
            'security_sentiment': "Security Sentiment",
            'legitimate_domain': "Legitimate Domain",
            'legitimate_headers': "Legitimate Headers",
            'legitimate_security': "Legitimate Security",
            'legitimate_formatting': "Legitimate Formatting",
            'legitimate_links': "Legitimate Links",
            'legitimate_language': "Legitimate Language",
            'legitimate_business': "Legitimate Business",
            'legitimate_marketing': "Legitimate Marketing",
            'legitimate_social': "Legitimate Social",
            'legitimate_technical': "Legitimate Technical"
        }
        
        # Enhanced threat level thresholds with confidence bands
        self.thresholds = {
            'critical': {'min': 0.95, 'confidence': 0.98},  # Increased threshold
            'high': {'min': 0.85, 'confidence': 0.90},     # Increased threshold
            'medium': {'min': 0.70, 'confidence': 0.80},   # Increased threshold
            'low': {'min': 0.50, 'confidence': 0.70},      # Increased threshold
            'safe': {'max': 0.30, 'confidence': 0.95}      # Increased confidence
        }
        
        # Initialize feature weights with focus on security indicators
        self.feature_weights = {
            'suspicious_urls': 0.25,           # Increased weight
            'malicious_attachments': 0.25,     # Increased weight
            'spoofed_headers': 0.15,           # Same weight
            'sensitive_keywords': 0.10,        # Decreased weight
            'security_indicators': 0.10,       # Same weight
            'social_engineering': 0.10,        # Same weight
            'security_grammar': 0.03,          # Decreased weight
            'security_sentiment': 0.02         # Decreased weight
        }
        
        # Initialize behavioral patterns
        self.behavioral_patterns = {
            'phishing_indicators': re.compile(r'(?i)(?:urgent|immediately|verify|confirm|update|security|alert|warning|suspicious|compromised)'),
            'legitimate_behavior': re.compile(r'(?i)(?:order\s*confirmation|shipping\s*notification|delivery\s*update|tracking\s*information)'),
            'suspicious_timing': re.compile(r'(?i)(?:within\s*24\s*hours|immediate\s*action|limited\s*time|expires\s*soon)'),
            'legitimate_timing': re.compile(r'(?i)(?:estimated\s*delivery|shipping\s*date|arrival\s*date|processing\s*time)')
        }
        
        # Initialize legitimate email verification patterns
        self.legitimate_patterns = {
            'legitimate_domains': re.compile(r'@(microsoft\.com|google\.com|apple\.com|amazon\.com|paypal\.com|linkedin\.com|facebook\.com|twitter\.com|github\.com|slack\.com|dropbox\.com|zoom\.us|atlassian\.net)$'),
            'legitimate_headers': re.compile(r'(?i)(?:from|reply-to|return-path|sender):\s*[^@]+@(microsoft\.com|google\.com|apple\.com|amazon\.com|paypal\.com|linkedin\.com|facebook\.com|twitter\.com|github\.com|slack\.com|dropbox\.com|zoom\.us|atlassian\.net)'),
            'legitimate_security': re.compile(r'(?i)(?:security\s*alert|account\s*activity|sign-in\s*notification|login\s*attempt|new\s*device|unusual\s*activity|two-factor\s*authentication|verification\s*code|password\s*reset)'),
            'legitimate_formatting': re.compile(r'(?i)(?:<html>.*?</html>|<body>.*?</body>|<p>.*?</p>|<ul>.*?</ul>|<li>.*?</li>|<div>.*?</div>|<span>.*?</span>|<a\s+href="[^"]*">.*?</a>)'),
            'legitimate_links': re.compile(r'https?://(?:www\.)?(?:microsoft\.com|google\.com|apple\.com|amazon\.com|paypal\.com|linkedin\.com|facebook\.com|twitter\.com|github\.com|slack\.com|dropbox\.com|zoom\.us|atlassian\.net)'),
            'legitimate_language': re.compile(r'(?i)(?:if this was you|you can safely disregard|if you did not|please contact support|for security reasons|to protect your account|this is an automated message|do not reply to this email)'),
            'legitimate_business': re.compile(r'(?i)(?:invoice|receipt|order\s*confirmation|shipping\s*notification|package\s*delivery|subscription\s*renewal|payment\s*confirmation|transaction\s*receipt)'),
            'legitimate_marketing': re.compile(r'(?i)(?:newsletter|promotion|special\s*offer|sale|discount|membership|loyalty\s*program|exclusive\s*deal|limited\s*time\s*offer)'),
            'legitimate_social': re.compile(r'(?i)(?:connection\s*request|friend\s*request|message\s*request|profile\s*view|endorsement|recommendation|network\s*update|activity\s*notification)'),
            'legitimate_technical': re.compile(r'(?i)(?:system\s*notification|service\s*update|maintenance\s*notice|outage\s*alert|performance\s*report|security\s*patch|version\s*update|bug\s*fix)')
        }
        
        # Initialize context patterns
        self.context_patterns = {
            'business_context': re.compile(r'(?i)(?:invoice|receipt|order|shipping|delivery|subscription|payment|transaction|account|statement)'),
            'security_context': re.compile(r'(?i)(?:security|account|login|password|verification|authentication|authorization|access|permission|privilege)'),
            'marketing_context': re.compile(r'(?i)(?:newsletter|promotion|offer|sale|discount|membership|loyalty|deal|special|limited)'),
            'social_context': re.compile(r'(?i)(?:connection|friend|message|profile|endorsement|recommendation|network|activity|update)'),
            'technical_context': re.compile(r'(?i)(?:system|service|maintenance|outage|performance|security|patch|version|bug|fix)')
        }
        
        # Initialize false positive prevention patterns
        self.false_positive_patterns = {
            'legitimate_urgency': re.compile(r'(?i)(?:urgent\s*security\s*update|important\s*account\s*notice|critical\s*system\s*alert|required\s*action|mandatory\s*update)'),
            'legitimate_requests': re.compile(r'(?i)(?:please\s*verify|confirm\s*your\s*identity|update\s*your\s*information|review\s*your\s*settings|check\s*your\s*preferences)'),
            'legitimate_actions': re.compile(r'(?i)(?:click\s*here\s*to\s*verify|follow\s*this\s*link|update\s*now|confirm\s*now|review\s*now)'),
            'legitimate_warnings': re.compile(r'(?i)(?:account\s*compromised|suspicious\s*activity|unauthorized\s*access|security\s*breach|data\s*exposure)')
        }
        
        # Initialize scoring weights
        self.scoring_weights = {
            'domain_auth': 0.35,    # Domain authentication weight
            'content': 0.25,        # Content analysis weight
            'behavior': 0.20,       # Behavioral analysis weight
            'context': 0.20         # Contextual analysis weight
        }
        
        # Initialize domain authentication patterns
        self.domain_auth_patterns = {
            'spf': re.compile(r'v=spf1.*'),
            'dkim': re.compile(r'v=DKIM1.*'),
            'dmarc': re.compile(r'v=DMARC1.*'),
            'legitimate_domains': {
                'amazon.com': {
                    'spf': 'v=spf1 include:spf1.amazon.com include:spf2.amazon.com ~all',
                    'dkim': 'v=DKIM1; k=rsa; p=',
                    'dmarc': 'v=DMARC1; p=reject; rua=mailto:dmarc@amazon.com'
                },
                'microsoft.com': {
                    'spf': 'v=spf1 include:spf.protection.outlook.com ~all',
                    'dkim': 'v=DKIM1; k=rsa; p=',
                    'dmarc': 'v=DMARC1; p=reject; rua=mailto:dmarc@microsoft.com'
                }
            }
        }

    def _verify_domain_authentication(self, email_text: str) -> dict:
        """Verify domain authentication using SPF, DKIM, and DMARC."""
        try:
            # Extract domain from email
            domain_match = re.search(r'@([^>]+)', email_text)
            if not domain_match:
                return {'score': 0.0, 'details': 'No domain found'}
            
            domain = domain_match.group(1)
            
            # Check if domain is in legitimate domains list
            if domain in self.legitimate_patterns['legitimate_domains']:
                expected_auth = self.legitimate_patterns['legitimate_domains'][domain]
                
                # Check SPF
                spf_match = expected_auth['spf'].search(email_text)
                spf_valid = spf_match and expected_auth['spf'] in spf_match.group(0)
                
                # Check DKIM
                dkim_match = expected_auth['dkim'].search(email_text)
                dkim_valid = dkim_match and expected_auth['dkim'] in dkim_match.group(0)
                
                # Check DMARC
                dmarc_match = expected_auth['dmarc'].search(email_text)
                dmarc_valid = dmarc_match and expected_auth['dmarc'] in dmarc_match.group(0)
                
                # Calculate authentication score
                auth_score = (spf_valid + dkim_valid + dmarc_valid) / 3.0
                
                return {
                    'score': auth_score,
                    'details': {
                        'spf': spf_valid,
                        'dkim': dkim_valid,
                        'dmarc': dmarc_valid
                    }
                }
            
            return {'score': 0.0, 'details': 'Domain not in legitimate list'}
            
        except Exception as e:
            logger.error(f"Error in domain authentication: {str(e)}")
            return {'score': 0.0, 'details': f'Error: {str(e)}'}

    def _analyze_content_structure(self, email_text: str) -> dict:
        """Analyze email content structure."""
        try:
            # Check HTML structure
            html_valid = bool(self.legitimate_patterns['legitimate_formatting'].search(email_text))
            
            # Check formatting
            formatting_valid = bool(self.legitimate_patterns['legitimate_formatting'].search(email_text))
            
            # Check links
            links = self.legitimate_patterns['legitimate_links'].findall(email_text)
            legitimate_links = all(any(re.search(pattern, link) for pattern in self.legitimate_patterns.values() if pattern != self.legitimate_patterns['legitimate_links']) for link in links)
            
            # Check headers
            headers_valid = bool(self.legitimate_patterns['legitimate_headers'].search(email_text))
            
            # Check content
            content_valid = bool(self.legitimate_patterns['legitimate_business'].search(email_text))
            
            # Calculate content score
            content_score = sum([
                html_valid,
                formatting_valid,
                legitimate_links,
                headers_valid,
                content_valid
            ]) / 5.0
            
            return {
                'score': content_score,
                'details': {
                    'html': html_valid,
                    'formatting': formatting_valid,
                    'links': legitimate_links,
                    'headers': headers_valid,
                    'content': content_valid
                }
            }
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            return {'score': 0.0, 'details': f'Error: {str(e)}'}

    def _analyze_behavior(self, email_text: str) -> dict:
        """Analyze email behavior patterns."""
        try:
            # Check for phishing indicators
            phishing_count = len(self.behavioral_patterns['phishing_indicators'].findall(email_text))
            
            # Check for legitimate behavior
            legitimate_count = len(self.behavioral_patterns['legitimate_behavior'].findall(email_text))
            
            # Check timing patterns
            suspicious_timing = len(self.behavioral_patterns['suspicious_timing'].findall(email_text))
            legitimate_timing = len(self.behavioral_patterns['legitimate_timing'].findall(email_text))
            
            # Calculate behavior score
            total_indicators = phishing_count + legitimate_count + suspicious_timing + legitimate_timing
            if total_indicators == 0:
                return {'score': 0.5, 'details': 'No behavioral patterns found'}
            
            behavior_score = (legitimate_count + legitimate_timing) / total_indicators
            
            return {
                'score': behavior_score,
                'details': {
                    'phishing_indicators': phishing_count,
                    'legitimate_behavior': legitimate_count,
                    'suspicious_timing': suspicious_timing,
                    'legitimate_timing': legitimate_timing
                }
            }
            
        except Exception as e:
            logger.error(f"Error in behavior analysis: {str(e)}")
            return {'score': 0.0, 'details': f'Error: {str(e)}'}

    def _analyze_context(self, email_text: str) -> dict:
        """Analyze email context."""
        try:
            # Check business context
            business_context = len(self.context_patterns['business_context'].findall(email_text))
            
            # Check security context
            security_context = len(self.context_patterns['security_context'].findall(email_text))
            
            # Check marketing context
            marketing_context = len(self.context_patterns['marketing_context'].findall(email_text))
            
            # Check social context
            social_context = len(self.context_patterns['social_context'].findall(email_text))
            
            # Calculate context score
            total_context = business_context + security_context + marketing_context + social_context
            if total_context == 0:
                return {'score': 0.5, 'details': 'No contextual patterns found'}
            
            context_score = business_context / total_context  # Prioritize business context
            
            return {
                'score': context_score,
                'details': {
                    'business': business_context,
                    'security': security_context,
                    'marketing': marketing_context,
                    'social': social_context
                }
            }
            
        except Exception as e:
            logger.error(f"Error in context analysis: {str(e)}")
            return {'score': 0.0, 'details': f'Error: {str(e)}'}

    def _calculate_rule_based_score(self, features: Dict[str, float], text: str) -> tuple:
        """Calculate phishing probability and threat level using multi-layer analysis."""
        try:
            # Perform multi-layer analysis
            domain_auth = self._verify_domain_authentication(text)
            content_analysis = self._analyze_content_structure(text)
            behavior_analysis = self._analyze_behavior(text)
            context_analysis = self._analyze_context(text)
            
            # Calculate base score from suspicious features
            base_score = sum(
                features[feature] * weight 
                for feature, weight in self.feature_weights.items()
                if feature in features
            )
            
            # Calculate legitimate adjustment based on legitimate features
            legitimate_features = [
                'legitimate_domain',
                'legitimate_headers',
                'legitimate_security',
                'legitimate_formatting',
                'legitimate_links',
                'legitimate_language',
                'legitimate_business',
                'legitimate_marketing',
                'legitimate_social',
                'legitimate_technical'
            ]
            
            # Calculate legitimate score (higher is better)
            legitimate_score = sum(
                features[feature] 
                for feature in legitimate_features 
                if feature in features
            ) / len(legitimate_features)
            
            # Apply legitimate adjustment to reduce threat score
            # Higher legitimate score means lower threat
            legitimate_adjustment = 1.0 - (legitimate_score * 0.6)  # 0.6 is the maximum reduction
            
            # Calculate final score with legitimate adjustment
            final_score = base_score * legitimate_adjustment
            
            # Determine threat level based on final score
            if final_score >= self.thresholds['critical']['min']:
                threat_level = 'Critical'
                confidence = self.thresholds['critical']['confidence']
            elif final_score >= self.thresholds['high']['min']:
                threat_level = 'High'
                confidence = self.thresholds['high']['confidence']
            elif final_score >= self.thresholds['medium']['min']:
                threat_level = 'Medium'
                confidence = self.thresholds['medium']['confidence']
            elif final_score >= self.thresholds['low']['min']:
                threat_level = 'Low'
                confidence = self.thresholds['low']['confidence']
            else:
                threat_level = 'Safe'
                confidence = self.thresholds['safe']['confidence']
            
            # Determine primary context
            context_details = context_analysis['details']
            primary_context = max(context_details.items(), key=lambda x: x[1])[0]
            
            return final_score, threat_level, confidence, primary_context
            
        except Exception as e:
            logger.error(f"Error in rule-based scoring: {str(e)}")
            return 0.0, 'Safe', 0.9, 'unknown'

    def predict(self, email_text: str) -> tuple:
        """Predict if email is phishing based on extracted features."""
        try:
            # Extract features
            features = self.extract_features(email_text)
            
            if self.model:
                # Get model prediction
                try:
                    # Try to get feature names from model
                    if hasattr(self.model, 'feature_names_in_'):
                        model_features = self.model.feature_names_in_
                        # Create feature vector with zeros for missing features
                        feature_values = []
                        for name in model_features:
                            # Map old feature names to new ones if needed
                            if name in features:
                                feature_values.append(features[name])
                            elif name == 'textblob_sentiment':
                                feature_values.append(features.get('sentiment', 0.0))
                            elif name == 'textblob_subjectivity':
                                feature_values.append(0.0)  # No longer used
                            elif name == 'verb_ratio':
                                feature_values.append(features.get('verb_noun_ratio', 0.0))
                            elif name == 'noun_ratio':
                                feature_values.append(0.0)  # No longer used
                            elif name == 'adj_ratio':
                                feature_values.append(0.0)  # No longer used
                            elif name == 'avg_sentence_length':
                                feature_values.append(0.0)  # No longer used
                            elif name == 'avg_word_length':
                                feature_values.append(0.0)  # No longer used
                            else:
                                feature_values.append(0.0)  # Default for unknown features
                        
                        probability = self.model.predict_proba([feature_values])[0][1]
                        # Convert probability to threat level
                        if probability >= 0.9:
                            threat_level = 'Safe'
                        elif probability >= 0.7:
                            threat_level = 'Low'
                        elif probability >= 0.5:
                            threat_level = 'Medium'
                        elif probability >= 0.3:
                            threat_level = 'High'
                        else:
                            threat_level = 'Critical'
                    else:
                        # Fallback to our feature weights if model doesn't have feature names
                        probability, threat_level, confidence, primary_context = self._calculate_rule_based_score(features, email_text)
                except Exception as e:
                    logger.warning(f"Error in model prediction, falling back to rule-based: {str(e)}")
                    probability, threat_level, confidence, primary_context = self._calculate_rule_based_score(features, email_text)
            else:
                # Rule-based fallback prediction
                probability, threat_level, confidence, primary_context = self._calculate_rule_based_score(features, email_text)
            
            return probability, threat_level, confidence, primary_context
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return 0.0, 'Safe', 0.9, 'unknown'

    def analyze_email(self, file_path: str) -> dict:
        """Analyze email file for phishing indicators."""
        try:
            # Read email content
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.endswith('.eml'):
                with open(file_path, 'rb') as f:
                    msg = BytesParser(policy=policy.default).parse(f)
                    text_content = []
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            text_content.append(part.get_content())
                        elif part.get_content_type() == 'text/html':
                            soup = BeautifulSoup(part.get_content(), 'html.parser')
                            text_content.append(soup.get_text())
                    email_text = '\n'.join(text_content)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    email_text = f.read()

            # Extract features first
            features = self.extract_features(email_text)
            
            # Get prediction
            probability, threat_level, confidence, primary_context = self.predict(email_text)

            # Determine phishing status
            is_phishing = probability > 0.5 or sum(features.values()) >= len(features) * 0.3

            result = {
                'label': 'Phishing' if is_phishing else 'Legitimate',
                'confidence': probability,
                'threat_level': threat_level,
                'confidence_band': confidence,
                'primary_context': primary_context,
                'indicators': features
            }

            self._display_results(result)
            return result

        except Exception as e:
            logger.error(f"Error analyzing email: {e}")
            console.print(f"[red]Error:[/red] {e}")
            return {'error': str(e)}

    def _display_results(self, result: dict):
        """Display analysis results in a user-friendly format."""
        console.print("\n[bold cyan]📧 Email Analysis Results[/bold cyan]")
        
        # Show threat level with appropriate color
        threat_level = result.get('threat_level', 'Safe')
        if threat_level == 'Critical':
            console.print(f"[bold red]⚠️  Threat Level: {threat_level}[/bold red]")
        elif threat_level == 'High':
            console.print(f"[bold yellow]⚠️  Threat Level: {threat_level}[/bold yellow]")
        elif threat_level == 'Medium':
            console.print(f"[bold orange]⚠️  Threat Level: {threat_level}[/bold orange]")
        elif threat_level == 'Low':
            console.print(f"[bold green]⚠️  Threat Level: {threat_level}[/bold green]")
        else:
            console.print(f"[bold green]✅ Threat Level: {threat_level}[/bold green]")
        
        # Show confidence score
        conf_percentage = result['confidence'] * 100
        conf_color = "red" if conf_percentage > 50 else "green"
        console.print(f"Confidence: [bold {conf_color}]{conf_percentage:.1f}%[/bold {conf_color}]")
        
        # Show confidence band
        conf_band = result.get('confidence_band', 0.0) * 100
        console.print(f"Confidence Band: [bold cyan]{conf_band:.1f}%[/bold cyan]")
        
        # Show primary context
        context = result.get('primary_context', 'unknown').capitalize()
        console.print(f"Primary Context: [bold cyan]{context}[/bold cyan]")

        # Show detected indicators
        if result['indicators']:
            console.print("\n[bold cyan]🔍 Detected Indicators[/bold cyan]")
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("Indicator", width=30)
            table.add_column("Status", justify="center", width=15)
            
            # Sort indicators by their values
            sorted_indicators = sorted(
                result['indicators'].items(),
                key=lambda x: (x[1] if isinstance(x[1], (int, float)) else 0),
                reverse=True
            )
            
            for feature, value in sorted_indicators:
                if value:  # Only show positive indicators
                    display_name = self.feature_names.get(feature, feature)
                    if isinstance(value, bool):
                        status = "🔴 Yes" if value else "🟢 No"
                    elif isinstance(value, (int, float)):
                        if value > 0.7:
                            status = "🔴 High"
                        elif value > 0.5:
                            status = "🟡 Medium"
                        elif value > 0.3:
                            status = "🟠 Low"
                        else:
                            status = "🟢 Safe"
                    else:
                        status = str(value)
                    
                    table.add_row(display_name, status)
            
        console.print(table)

    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def save_model_metrics(self, y_true, y_pred, y_probs, output_dir: str):
        """Save model evaluation metrics and visualizations."""
        try:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
            pr_auc = auc(recall, precision)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
            plt.close()

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                plt.figure(figsize=(10, 6))
                indices = np.argsort(feature_importance)[::-1]
                plt.title('Feature Importance')
                plt.bar(range(len(feature_importance)), feature_importance[indices])
                plt.xticks(range(len(feature_importance)), 
                          [self.feature_names.keys()[i] for i in indices], 
                          rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()

            # Save metrics to file
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'log_loss': log_loss(y_true, y_probs),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Model metrics and visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model metrics: {str(e)}")
            console.print(f"[red]Error saving model metrics: {str(e)}[/red]")

    def extract_features(self, text: str) -> dict:
        """Extract and analyze features from email text."""
        try:
            # Initialize features dictionary
            features = {
                'suspicious_urls': 0.0,
                'malicious_attachments': 0.0,
                'spoofed_headers': 0.0,
                'sensitive_keywords': 0.0,
                'security_indicators': 0.0,
                'social_engineering': 0.0,
                'security_grammar': 0.0,
                'security_sentiment': 0.0,
                'legitimate_domain': 0.0,
                'legitimate_headers': 0.0,
                'legitimate_security': 0.0,
                'legitimate_formatting': 0.0,
                'legitimate_links': 0.0,
                'legitimate_language': 0.0,
                'legitimate_business': 0.0,
                'legitimate_marketing': 0.0,
                'legitimate_social': 0.0,
                'legitimate_technical': 0.0
            }
            
            # Domain Authentication Analysis
            domain_auth = self._verify_domain_authentication(text)
            features['legitimate_domain'] = domain_auth['score']
            
            # Content Structure Analysis
            content_analysis = self._analyze_content_structure(text)
            features['legitimate_formatting'] = content_analysis['score']
            
            # Check for legitimate links
            links = self.legitimate_patterns['legitimate_links'].findall(text)
            if links:
                legitimate_links = sum(1 for link in links if any(re.search(pattern, link) for pattern in self.legitimate_patterns.values() if pattern != self.legitimate_patterns['legitimate_links']))
                features['legitimate_links'] = legitimate_links / len(links)
            
            # Check for legitimate headers
            if self.legitimate_patterns['legitimate_headers'].search(text):
                features['legitimate_headers'] = 1.0
            
            # Behavioral Analysis
            behavior_analysis = self._analyze_behavior(text)
            
            # Check for phishing indicators
            phishing_count = len(self.behavioral_patterns['phishing_indicators'].findall(text))
            if phishing_count > 0:
                features['social_engineering'] = min(1.0, phishing_count / 5.0)
            
            # Check for legitimate behavior
            legitimate_count = len(self.behavioral_patterns['legitimate_behavior'].findall(text))
            if legitimate_count > 0:
                features['legitimate_business'] = min(1.0, legitimate_count / 5.0)
            
            # Context Analysis
            context_analysis = self._analyze_context(text)
            
            # Check business context
            business_context = len(self.context_patterns['business_context'].findall(text))
            if business_context > 0:
                features['legitimate_business'] = max(features['legitimate_business'], min(1.0, business_context / 5.0))
            
            # Check security context
            security_context = len(self.context_patterns['security_context'].findall(text))
            if security_context > 0:
                features['legitimate_security'] = min(1.0, security_context / 5.0)
            
            # Check for sensitive keywords
            sensitive_keywords = len(self.behavioral_patterns['phishing_indicators'].findall(text))
            if sensitive_keywords > 0:
                features['sensitive_keywords'] = min(1.0, sensitive_keywords / 5.0)
            
            # Check for security indicators
            security_indicators = len(self.behavioral_patterns['legitimate_behavior'].findall(text))
            if security_indicators > 0:
                features['security_indicators'] = min(1.0, security_indicators / 5.0)
            
            # Check for legitimate language
            legitimate_language = len(self.behavioral_patterns['legitimate_timing'].findall(text))
            if legitimate_language > 0:
                features['legitimate_language'] = min(1.0, legitimate_language / 5.0)
            
            # Check for suspicious URLs
            suspicious_urls = sum(1 for link in links if not any(re.search(pattern, link) for pattern in self.legitimate_patterns.values() if pattern != self.legitimate_patterns['legitimate_links']))
            if suspicious_urls > 0:
                features['suspicious_urls'] = min(1.0, suspicious_urls / len(links))
            
            # Check for malicious attachments
            if re.search(r'\.(exe|scr|bat|cmd|msi|dll|vbs|js|wsf|hta|ps1)$', text, re.IGNORECASE):
                features['malicious_attachments'] = 1.0
            
            # Check for spoofed headers
            if re.search(r'(?i)(?:from|reply-to|return-path|sender):\s*[^@]+@[^@]+\.[^@]+', text):
                features['spoofed_headers'] = 1.0
            
            # Check for security grammar
            security_grammar = len(self.behavioral_patterns['phishing_indicators'].findall(text))
            if security_grammar > 0:
                features['security_grammar'] = min(1.0, security_grammar / 5.0)
            
            # Check for security sentiment
            security_sentiment = len(self.behavioral_patterns['suspicious_timing'].findall(text))
            if security_sentiment > 0:
                features['security_sentiment'] = min(1.0, security_sentiment / 5.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {name: 0.0 for name in features.keys()}

def train_from_csv(model_path: str, csv_path: str, config_path: str = CONFIG_PATH):
    """Train the model from a CSV file with improved metrics and visualization."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss, precision_recall_curve, roc_curve, auc
    import torch

    class EmailDataset:
        def __init__(self, emails, labels, feature_extractor):
            self.emails = emails
            self.labels = labels
            self.feature_extractor = feature_extractor
            self.feature_names = feature_extractor.feature_names.keys()

        def __len__(self):
            return len(self.emails)

        def __getitem__(self, idx):
            email = str(self.emails[idx])
            label = self.labels[idx]
            
            # Extract features
            feature_dict = self.feature_extractor.extract_features(email)
            features = [feature_dict[name] for name in self.feature_names]
            
            return torch.FloatTensor(features), torch.LongTensor([label])

    try:
        # Load configuration
        config = ModelConfig()
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config = ModelConfig(**config_dict)

        # Load and validate data
        console.print("[cyan]Loading dataset...[/cyan]")
        df = pd.read_csv(csv_path)
        if 'email_content' not in df.columns or 'label' not in df.columns:
            console.print("[bold red]CSV must contain 'email_content' and 'label' columns.[/bold red]")
            return

        # Create feature extractor
        feature_extractor = PhishingDetector(model_path=None)
        
        # Split data
        console.print("[cyan]Splitting data into train and validation sets...[/cyan]")
        train_df, val_df = train_test_split(
            df, 
            test_size=config.test_size, 
            random_state=config.random_state
        )

        # Create datasets
        train_dataset = EmailDataset(
            train_df['email_content'].values,
            train_df['label'].values,
            feature_extractor
        )
        val_dataset = EmailDataset(
            val_df['email_content'].values,
            val_df['label'].values,
            feature_extractor
        )

        # Process data in batches
        console.print("[cyan]Extracting features in batches...[/cyan]")
        batch_size = 32  # Adjust based on your memory
        X_train = []
        y_train = []
        X_val = []
        y_val = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            # Process training data
            task = progress.add_task("Processing training data...", total=len(train_dataset))
            for i in range(0, len(train_dataset), batch_size):
                batch_end = min(i + batch_size, len(train_dataset))
                batch_features, batch_labels = zip(*[train_dataset[j] for j in range(i, batch_end)])
                X_train.extend([f.numpy() for f in batch_features])
                y_train.extend([l.numpy() for l in batch_labels])
                progress.update(task, completed=batch_end)

            # Process validation data
            task = progress.add_task("Processing validation data...", total=len(val_dataset))
            for i in range(0, len(val_dataset), batch_size):
                batch_end = min(i + batch_size, len(val_dataset))
                batch_features, batch_labels = zip(*[val_dataset[j] for j in range(i, batch_end)])
                X_val.extend([f.numpy() for f in batch_features])
                y_val.extend([l.numpy() for l in batch_labels])
                progress.update(task, completed=batch_end)

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train).ravel()
        X_val = np.array(X_val)
        y_val = np.array(y_val).ravel()

        # Train model
        console.print("[cyan]Training model...[/cyan]")
        clf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            verbose=1,
            n_jobs=-1  # Use all available cores
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training...", total=None)
            clf.fit(X_train, y_train)
            progress.update(task, completed=True)

        # Make predictions
        console.print("[cyan]Evaluating model...[/cyan]")
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, preds)
        loss = log_loss(y_val, probs)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_val, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_val, probs[:, 1])
        pr_auc = auc(recall, precision)

        # Display metrics
        console.print("\n[bold green]Training Results:[/bold green]")
        console.print(f"✅ Accuracy: {acc:.4f}")
        console.print(f"✅ Log Loss: {loss:.4f}")
        console.print(f"✅ ROC AUC: {roc_auc:.4f}")
        console.print(f"✅ PR AUC: {pr_auc:.4f}")

        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            joblib.dump(clf, model_path)
            console.print(f"[green]✅ Model saved to {model_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving model: {str(e)}[/red]")
            return
        
        # Create visualizations directory
        viz_dir = os.path.join(os.path.dirname(model_path), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Save model metrics and visualizations
        feature_extractor.model = clf  # Set the trained model
        feature_extractor.save_model_metrics(y_val, preds, probs, viz_dir)
        
        console.print(f"[green]✅ Visualizations saved to {viz_dir}[/green]")
        console.print("\n[bold green]Training completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]Error during training: {str(e)}[/red]")
        logger.error(f"Training error: {str(e)}")
        logger.error(traceback.format_exc())

def batch_predict(model: PhishingDetector, folder: str, parallel: bool = True):
    """Process multiple email files with optional parallel processing."""
    if not os.path.isdir(folder):
        console.print("[red]Folder not found.[/red]")
        return

    files = [f for f in os.listdir(folder) if f.endswith(('.eml', '.txt'))]
    if not files:
        console.print("[yellow]No .eml or .txt files found.[/yellow]")
        return

    table = Table(title="📧 Batch Email Predictions")
    table.add_column("File", style="cyan")
    table.add_column("Prediction", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Processing Time", justify="center")

    def process_file(file: str) -> tuple:
        start_time = datetime.now()
        try:
            path = os.path.join(folder, file)
            result = model.analyze_email(path)
            processing_time = (datetime.now() - start_time).total_seconds()
            return file, result['label'], f"{result['confidence']:.2f}", f"{processing_time:.2f}s"
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return file, "[red]Error[/red]", "-", f"{processing_time:.2f}s"

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_file, f) for f in files]
            for future in concurrent.futures.as_completed(futures):
                file, label, conf, time = future.result()
                table.add_row(file, label, conf, time)
    else:
        for f in files:
            file, label, conf, time = process_file(f)
            table.add_row(file, label, conf, time)

    console.print(table)
    
def main():
    """Main CLI interface with enhanced options."""
    try:
        # Ensure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        model = PhishingDetector(MODEL_PATH)
        while True:
            console.print("\n[bold yellow]Phishing Email CLI[/bold yellow]")
            console.print("[1] Predict email file")
            console.print("[2] Batch predict from folder")
            console.print("[3] Train model from CSV")
            console.print("[4] Show model statistics")
            console.print("[5] Configure model")
            console.print("[6] Exit")
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6"], default="1")
            
            if choice == "1":
                path = Prompt.ask("Enter path to email (.eml/.txt) file")
                if os.path.exists(path):
                    try:
                        model.analyze_email(path)
                    except Exception as e:
                        console.print(f"[red]Error:[/red] {e}")
                else:
                    console.print("[red]File not found.[/red]")
            elif choice == "2":
                folder = Prompt.ask("Enter folder path containing email files")
                parallel = Prompt.ask("Use parallel processing?", choices=["y", "n"], default="y") == "y"
                batch_predict(model, folder, parallel)
            elif choice == "3":
                if os.path.exists(DATASET_PATH):
                    console.print(f"[cyan]Training model using dataset: {DATASET_PATH}[/cyan]")
                    train_from_csv(MODEL_PATH, DATASET_PATH)
                else:
                    console.print(f"[red]Error: Dataset file not found at {DATASET_PATH}[/red]")
            elif choice == "4":
                if model.model is not None:
                    model.plot_feature_importance()
                    console.print("[green]Feature importance plot displayed[/green]")
                else:
                    console.print("[red]No trained model available[/red]")
            elif choice == "5":
                config_path = Prompt.ask("Enter path to configuration file (or press Enter for default)")
                if config_path:
                    model = PhishingDetector(MODEL_PATH)
                    console.print("[green]Model reconfigured successfully[/green]")
            elif choice == "6":
                break
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_treebank_pos_tagger'])
    main()
