"""Core analysis engine for EconoLens AI.

Author: Rishabh Shah
Advisor: Dr. Qingyang Xiao
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import random
import re
import socket
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier

try:
    import trafilatura
except Exception:  # Optional dependency for robust webpage extraction.
    trafilatura = None


APP_NAME = "EconoLens AI"
APP_TAGLINE = "Economics News Simplifier and Personal Impact Storyteller"
STUDENT_NAME = "Rishabh Shah"
ADVISOR_NAME = "Dr. Qingyang Xiao"
MODEL_NAME = "google/flan-t5-small"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass(slots=True)
class ReaderProfile:
    country: str = "United States"
    age_group: str = "18-24"
    role: str = "Student"
    income_situation: str = "Middle income"
    housing_status: str = "Renter"
    debt_exposure: str = "Some student or consumer debt"
    budget_sensitivity: str = "Moderate"
    interests: tuple[str, ...] = (
        "everyday prices",
        "jobs",
        "rent",
        "student loans",
    )

    @classmethod
    def from_interests_text(cls, interests_text: str, **kwargs: Any) -> "ReaderProfile":
        interests = tuple(
            item.strip()
            for item in re.split(r"[,;|]", interests_text)
            if item.strip()
        )
        return cls(interests=interests or cls().interests, **kwargs)


@dataclass(slots=True)
class ArticleMetadata:
    title: str = "Untitled economics article"
    source: str = "User-provided text"
    published_date: str = ""
    url: str = ""


ECONOMIC_GLOSSARY: dict[str, dict[str, str]] = {
    "aggregate demand": {
        "plain": "total spending by households, businesses, government, and foreign buyers",
        "life": "It helps explain whether businesses are likely to sell more, hire more, or raise prices.",
    },
    "aggregate supply": {
        "plain": "the total amount of goods and services the economy can produce",
        "life": "When supply cannot keep up with demand, shortages and price pressure can appear.",
    },
    "balance of trade": {
        "plain": "the difference between what a country exports and imports",
        "life": "It can influence industries, exchange rates, and the price of imported products.",
    },
    "benchmark interest rate": {
        "plain": "a key rate set or guided by a central bank that influences many other rates",
        "life": "It can affect credit cards, car loans, mortgages, savings accounts, and business borrowing.",
    },
    "bond yield": {
        "plain": "the return investors expect from holding a bond",
        "life": "Bond yields often influence mortgage rates, government borrowing costs, and investment choices.",
    },
    "budget deficit": {
        "plain": "the amount by which government spending exceeds government revenue in a period",
        "life": "Persistent deficits can affect taxes, public services, government borrowing, and interest costs.",
    },
    "business cycle": {
        "plain": "the recurring pattern of economic expansion, slowdown, recession, and recovery",
        "life": "Different phases can change hiring, wages, business sales, and household confidence.",
    },
    "central bank": {
        "plain": "the institution that manages monetary policy and helps oversee the financial system",
        "life": "Its decisions can influence borrowing costs, inflation, employment, and financial markets.",
    },
    "consumer confidence": {
        "plain": "a measure of how optimistic or worried households feel about the economy",
        "life": "Confidence can influence whether people spend, save, delay purchases, or look for new jobs.",
    },
    "consumer price index": {
        "plain": "a measure of how prices paid by consumers change over time",
        "life": "It is commonly used to discuss inflation and changes in household purchasing power.",
    },
    "core inflation": {
        "plain": "inflation measured after excluding categories that often swing sharply, usually food and energy",
        "life": "It is used to judge whether broad price pressure may last, even when gasoline or food prices jump around.",
    },
    "credit spread": {
        "plain": "the extra interest investors demand for lending to a riskier borrower",
        "life": "Wider spreads can make it harder or more expensive for companies and some consumers to borrow.",
    },
    "current account": {
        "plain": "a broad record of a country's trade, income, and certain international payments",
        "life": "Large imbalances can influence currencies, trade debates, and dependence on foreign financing.",
    },
    "deflation": {
        "plain": "a broad decline in prices across the economy",
        "life": "Falling prices may sound helpful, but prolonged deflation can weaken wages, profits, hiring, and debt repayment.",
    },
    "disinflation": {
        "plain": "prices are still rising, but more slowly than before",
        "life": "Your costs may continue to increase even though the inflation rate is coming down.",
    },
    "disposable income": {
        "plain": "income left after taxes",
        "life": "It is the money available for housing, food, transportation, saving, debt, and optional spending.",
    },
    "economic contraction": {
        "plain": "a period when overall economic activity is shrinking",
        "life": "Businesses may sell less, reduce hours, slow hiring, or postpone investment.",
    },
    "economic expansion": {
        "plain": "a period when production, income, and employment are generally growing",
        "life": "It often supports hiring and wages, although prices and interest rates can also rise.",
    },
    "exchange rate": {
        "plain": "the value of one currency compared with another",
        "life": "It affects travel costs, imported goods, exports, and companies that earn money in other countries.",
    },
    "federal funds rate": {
        "plain": "the overnight interest-rate target used by the U.S. Federal Reserve",
        "life": "Changes can influence many borrowing and saving rates throughout the U.S. economy.",
    },
    "fiscal policy": {
        "plain": "government decisions about spending and taxes",
        "life": "It can change household benefits, public services, taxes, business demand, and the overall economy.",
    },
    "gross domestic product": {
        "plain": "the total value of final goods and services produced within a country",
        "life": "It is a broad measure of economic activity, but it does not show how gains or losses are shared.",
    },
    "headline inflation": {
        "plain": "overall inflation including categories such as food and energy",
        "life": "It often reflects the price changes households notice most directly.",
    },
    "inflation": {
        "plain": "a broad rise in prices over time",
        "life": "If income does not keep pace, the same paycheck buys less.",
    },
    "labor force participation rate": {
        "plain": "the share of working-age people who are employed or actively looking for work",
        "life": "It adds context to the unemployment rate and can reveal whether people are entering or leaving the job market.",
    },
    "liquidity": {
        "plain": "how easily money or an asset can be used, bought, or sold without a large price change",
        "life": "Low liquidity can make borrowing, selling investments, or meeting short-term obligations harder.",
    },
    "monetary policy": {
        "plain": "central-bank actions intended to influence inflation, employment, credit, and economic activity",
        "life": "It affects rates on borrowing and saving and can influence job and price conditions.",
    },
    "national debt": {
        "plain": "the accumulated amount a national government owes",
        "life": "Interest costs can compete with other public priorities and shape future tax and spending debates.",
    },
    "nominal wage": {
        "plain": "the dollar amount of pay before adjusting for inflation",
        "life": "A raise can look strong in dollars but still buy less if prices rise faster.",
    },
    "output gap": {
        "plain": "the difference between actual economic production and a sustainable potential level",
        "life": "A large positive gap can add inflation pressure, while a negative gap can signal unused workers and resources.",
    },
    "personal consumption expenditures": {
        "plain": "a broad measure of consumer spending and prices used in U.S. economic analysis",
        "life": "It helps policymakers evaluate inflation and changes in household spending.",
    },
    "producer price index": {
        "plain": "a measure of price changes received by producers",
        "life": "Higher business input prices can eventually reach consumers, though not always fully or immediately.",
    },
    "productivity": {
        "plain": "how much output is produced for each unit of work or other input",
        "life": "Sustained productivity growth can support higher wages and living standards without the same inflation pressure.",
    },
    "purchasing managers index": {
        "plain": "a survey-based signal of whether business activity is expanding or shrinking",
        "life": "It can offer an early clue about orders, production, jobs, and supply conditions.",
    },
    "purchasing power": {
        "plain": "how much your money can buy",
        "life": "Purchasing power falls when prices rise faster than your income or savings return.",
    },
    "quantitative easing": {
        "plain": "central-bank purchases of financial assets intended to lower borrowing costs and support the economy",
        "life": "It can influence mortgage rates, asset prices, lending conditions, and the value of cash savings.",
    },
    "quantitative tightening": {
        "plain": "a central bank reducing the financial assets it holds",
        "life": "It can remove support from financial markets and contribute to tighter borrowing conditions.",
    },
    "real interest rate": {
        "plain": "an interest rate after accounting for inflation",
        "life": "It helps show whether borrowing is truly expensive and whether savings are gaining purchasing power.",
    },
    "real wage": {
        "plain": "pay after adjusting for inflation",
        "life": "Real wage growth means earnings are increasing faster than prices.",
    },
    "recession": {
        "plain": "a broad, meaningful decline in economic activity",
        "life": "It can reduce hiring, hours, business income, and confidence, although experiences differ across industries and households.",
    },
    "retail sales": {
        "plain": "the value of goods sold by retailers",
        "life": "It helps indicate how strongly households are spending and which types of businesses may be gaining or losing demand.",
    },
    "risk premium": {
        "plain": "extra expected return demanded for taking more risk",
        "life": "A higher risk premium can raise financing costs and reduce the value of riskier investments.",
    },
    "soft landing": {
        "plain": "inflation slows without a severe recession or major job losses",
        "life": "It would mean price pressure eases while employment and incomes remain relatively stable.",
    },
    "stagflation": {
        "plain": "high inflation combined with weak growth and a difficult job market",
        "life": "Households can face rising costs at the same time that earnings and job opportunities weaken.",
    },
    "supply chain": {
        "plain": "the network that moves materials and products from suppliers to buyers",
        "life": "Disruptions can create delays, shortages, or higher prices for groceries, vehicles, electronics, and other goods.",
    },
    "tariff": {
        "plain": "a tax placed on imported goods",
        "life": "Some of the cost may be paid by importers, businesses, or consumers through higher prices or changed product choices.",
    },
    "tight labor market": {
        "plain": "a job market where employers have difficulty finding enough workers",
        "life": "Workers may have more bargaining power, but labor shortages can also raise business costs and prices.",
    },
    "unemployment rate": {
        "plain": "the share of people in the labor force who do not have a job and are actively looking",
        "life": "It is an important job-market measure, but it does not count everyone who wants more work or stopped searching.",
    },
    "yield curve": {
        "plain": "a comparison of interest rates on bonds with different maturities",
        "life": "Its shape can affect bank lending and is often watched for clues about growth and recession risk.",
    },
}


PLAIN_LANGUAGE_REPLACEMENTS: dict[str, str] = {
    "amid": "during",
    "approximately": "about",
    "bolster": "support",
    "commence": "start",
    "consequently": "as a result",
    "contraction": "decline",
    "deteriorate": "get worse",
    "elevated": "high",
    "exacerbate": "make worse",
    "facilitate": "help",
    "furthermore": "also",
    "headwinds": "forces making growth harder",
    "indicate": "show",
    "mitigate": "reduce",
    "moderate": "slow",
    "notwithstanding": "despite",
    "persist": "continue",
    "pressures": "problems or forces",
    "recalibrate": "adjust",
    "subsequently": "later",
    "substantial": "large",
    "trajectory": "direction",
    "transitory": "temporary",
    "underpin": "support",
    "utilize": "use",
}


PLAIN_TERM_REPLACEMENTS: dict[str, str] = {
    "annual consumer inflation": "annual consumer price growth",
    "consumer inflation": "consumer price growth",
    "aggregate demand": "total spending",
    "aggregate supply": "total production capacity",
    "benchmark interest rate": "key interest rate",
    "bond yield": "bond return",
    "budget deficit": "government budget shortfall",
    "central bank": "main national bank",
    "consumer confidence": "household confidence about the economy",
    "consumer price index": "consumer price measure",
    "core inflation": "underlying price growth",
    "credit spread": "extra borrowing risk cost",
    "deflation": "broadly falling prices",
    "disinflation": "slower price growth",
    "disposable income": "income after taxes",
    "economic contraction": "economic decline",
    "economic expansion": "economic growth period",
    "exchange rate": "currency value",
    "federal funds rate": "Federal Reserve key rate",
    "fiscal policy": "government tax and spending policy",
    "gross domestic product": "total economic output",
    "headline inflation": "overall price growth",
    "inflation": "overall price growth",
    "labor force participation rate": "share of adults working or job hunting",
    "monetary policy": "central-bank rate policy",
    "national debt": "total government debt",
    "nominal wage": "pay before adjusting for prices",
    "personal consumption expenditures": "consumer spending and price measure",
    "producer price index": "business selling-price measure",
    "purchasing managers index": "business activity survey",
    "purchasing power": "how much money can buy",
    "quantitative easing": "central-bank asset buying",
    "quantitative tightening": "central-bank balance-sheet reduction",
    "real interest rate": "interest rate after inflation",
    "real wage": "pay after inflation",
    "recession": "broad economic downturn",
    "retail sales": "store and online sales",
    "risk premium": "extra return for taking risk",
    "soft landing": "slower inflation without a major downturn",
    "stagflation": "high prices with weak growth",
    "supply chain": "product delivery network",
    "tariff": "import tax",
    "tight labor market": "job market with more openings than available workers",
    "unemployment rate": "share of job seekers without work",
    "yield curve": "comparison of short- and long-term bond rates",
}


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Inflation and prices": (
        "inflation",
        "consumer price",
        "price growth",
        "cost of living",
        "core inflation",
        "deflation",
        "disinflation",
        "producer prices",
        "purchasing power",
        "food prices",
        "energy prices",
    ),
    "Interest rates and credit": (
        "interest rate",
        "central bank",
        "federal reserve",
        "monetary policy",
        "credit card",
        "loan",
        "borrowing cost",
        "bond yield",
        "mortgage rate",
        "rate cut",
        "rate hike",
        "liquidity",
    ),
    "Jobs and wages": (
        "employment",
        "unemployment",
        "jobs",
        "hiring",
        "layoff",
        "wage",
        "salary",
        "labor market",
        "payroll",
        "job openings",
        "participation rate",
    ),
    "Growth and business activity": (
        "gross domestic product",
        "gdp",
        "growth",
        "recession",
        "expansion",
        "contraction",
        "productivity",
        "business investment",
        "factory output",
        "services activity",
        "purchasing managers",
    ),
    "Housing and real estate": (
        "housing",
        "home prices",
        "rent",
        "mortgage",
        "construction",
        "home sales",
        "housing starts",
        "property",
        "vacancy",
        "real estate",
    ),
    "Trade, currency, and supply chains": (
        "trade",
        "import",
        "export",
        "tariff",
        "exchange rate",
        "currency",
        "supply chain",
        "shipping",
        "shortage",
        "trade deficit",
        "port",
    ),
    "Government policy and public finance": (
        "government spending",
        "tax",
        "fiscal policy",
        "budget deficit",
        "national debt",
        "subsidy",
        "benefit",
        "regulation",
        "public investment",
        "treasury",
    ),
    "Markets and investing": (
        "stock market",
        "stocks",
        "bonds",
        "investor",
        "earnings",
        "volatility",
        "risk premium",
        "asset prices",
        "index",
        "portfolio",
    ),
    "Consumer spending and household finance": (
        "consumer spending",
        "retail sales",
        "household income",
        "disposable income",
        "saving rate",
        "credit balance",
        "debt payment",
        "consumer confidence",
        "personal finance",
        "household budget",
    ),
}


TOPIC_EXAMPLES: dict[str, list[str]] = {
    "Inflation and prices": [
        "Consumer prices rose again as food and service costs increased.",
        "Core inflation slowed, but rent and insurance remained expensive.",
        "The cost of living is climbing faster than many household paychecks.",
        "Producer prices fell, reducing some future pressure on store prices.",
        "Disinflation means prices are rising more slowly rather than falling.",
        "Higher gasoline and grocery prices reduced household purchasing power.",
        "The inflation report showed broad price pressure across services.",
        "Falling energy prices pulled the headline consumer price index lower.",
        "Businesses said input costs were increasing and margins were squeezed.",
        "Price growth eased as supply improved and demand cooled.",
    ],
    "Interest rates and credit": [
        "The central bank held its benchmark interest rate unchanged.",
        "A rate cut could lower some borrowing costs for households and firms.",
        "Credit card annual percentage rates remained high after monetary tightening.",
        "Bond yields rose as investors expected interest rates to stay high.",
        "Mortgage rates increased and reduced homebuyer affordability.",
        "Banks tightened loan standards for small businesses and consumers.",
        "The Federal Reserve signaled that future policy depends on inflation data.",
        "Savings account yields improved after the central bank raised rates.",
        "Lower market interest rates made refinancing more attractive.",
        "Liquidity conditions weakened and lenders demanded more compensation for risk.",
    ],
    "Jobs and wages": [
        "Employers added jobs, but hiring was concentrated in a few industries.",
        "The unemployment rate increased as more people looked for work.",
        "Wage growth slowed while payroll employment continued to expand.",
        "Layoff announcements rose in technology and manufacturing.",
        "Job openings fell, suggesting employers had less unmet demand for workers.",
        "A tight labor market helped workers negotiate higher salaries.",
        "Labor force participation improved among prime-age workers.",
        "Weekly hours declined even though the total number of jobs increased.",
        "Small businesses reported difficulty finding qualified employees.",
        "Real wages rose because pay increased faster than consumer prices.",
    ],
    "Growth and business activity": [
        "Gross domestic product expanded as services and business investment grew.",
        "Factory output contracted because new orders weakened.",
        "Economists raised the recession risk after several months of slower activity.",
        "Productivity gains allowed companies to produce more with the same labor.",
        "The purchasing managers index moved above the expansion threshold.",
        "Consumer demand supported economic growth despite weak exports.",
        "Business investment slowed as financing costs increased.",
        "The economy entered a broad contraction in production and income.",
        "A soft landing would combine lower inflation with continued growth.",
        "Services activity accelerated while manufacturing remained weak.",
    ],
    "Housing and real estate": [
        "Home prices increased even as mortgage rates remained high.",
        "Apartment rents slowed because more units became available.",
        "Housing starts rose as builders responded to limited home inventory.",
        "Existing home sales declined because owners were reluctant to move.",
        "Higher property taxes raised monthly costs for homeowners.",
        "Mortgage applications fell after borrowing rates increased.",
        "Vacancy rates rose in several office real estate markets.",
        "First-time buyers struggled with down payments and monthly affordability.",
        "Construction costs declined as material supply improved.",
        "Renters faced renewal increases that exceeded their wage gains.",
    ],
    "Trade, currency, and supply chains": [
        "New tariffs increased the cost of selected imported goods.",
        "The currency strengthened and made foreign travel less expensive.",
        "Port delays disrupted supply chains and slowed product deliveries.",
        "Exports weakened because demand from major trading partners declined.",
        "A shipping shortage raised freight rates for retailers and manufacturers.",
        "The trade deficit widened as imports grew faster than exports.",
        "A weaker currency increased the domestic price of imported energy.",
        "Companies moved suppliers to reduce exposure to geopolitical risk.",
        "Customs restrictions created shortages of important industrial inputs.",
        "Improved logistics reduced delivery times and inventory pressure.",
    ],
    "Government policy and public finance": [
        "The government proposed new tax credits for lower-income households.",
        "A larger budget deficit required additional government borrowing.",
        "Public infrastructure spending supported construction employment.",
        "Lawmakers debated reductions in benefits and discretionary programs.",
        "A subsidy lowered the consumer price of selected energy products.",
        "New regulation increased compliance costs for financial institutions.",
        "Tax revenue rose as wages and corporate profits increased.",
        "Interest payments consumed a larger share of the national budget.",
        "Fiscal policy became more restrictive as emergency programs expired.",
        "Local governments increased property taxes to fund public services.",
    ],
    "Markets and investing": [
        "Stock indexes fell as investors reassessed earnings expectations.",
        "Bond prices declined when market yields moved higher.",
        "Volatility increased after companies issued cautious forecasts.",
        "Investors shifted toward safer assets during a period of uncertainty.",
        "Corporate earnings grew, but profit margins narrowed.",
        "A lower risk premium lifted the valuation of growth stocks.",
        "Financial markets rallied after inflation came in below expectations.",
        "Long-term bond yields affected the value of rate-sensitive investments.",
        "Portfolio diversification reduced dependence on a single asset class.",
        "Bank shares weakened as credit losses increased.",
    ],
    "Consumer spending and household finance": [
        "Retail sales increased as households spent more on services.",
        "Consumer confidence weakened because families worried about jobs and prices.",
        "Disposable income grew after taxes, supporting household spending.",
        "The household saving rate declined as people used cash reserves.",
        "Credit balances rose and monthly debt payments took more income.",
        "Consumers delayed large purchases because financing was expensive.",
        "Higher utility bills forced families to reduce optional spending.",
        "Household income growth helped offset rising grocery costs.",
        "Delinquency rates increased for some auto and credit card borrowers.",
        "Budget-conscious consumers switched to lower-priced products.",
    ],
}


UP_WORDS = {
    "accelerated",
    "advanced",
    "climbed",
    "expanded",
    "grew",
    "higher",
    "hiked",
    "increased",
    "jumped",
    "raised",
    "rebounded",
    "rose",
    "strengthened",
    "surged",
    "up",
    "widened",
}
DOWN_WORDS = {
    "contracted",
    "cooled",
    "cut",
    "declined",
    "decreased",
    "dropped",
    "eased",
    "fell",
    "lower",
    "reduced",
    "slowed",
    "softened",
    "weakened",
    "down",
    "narrowed",
}
UNCERTAINTY_CUES = {
    "may",
    "might",
    "could",
    "uncertain",
    "risk",
    "forecast",
    "outlook",
    "expected",
    "projected",
    "scenario",
    "if",
}
CAUSAL_CUES = (
    "because",
    "due to",
    "driven by",
    "after",
    "as a result",
    "reflecting",
    "caused by",
    "owing to",
    "supported by",
    "weighed down by",
)
OUTLOOK_CUES = (
    "will",
    "may",
    "might",
    "could",
    "expected",
    "forecast",
    "outlook",
    "next",
    "future",
    "projected",
)

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
}


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_article_text(text: str, max_chars: int = 60000) -> tuple[str, list[str]]:
    warnings_list: list[str] = []
    text = normalize_space(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars]
        warnings_list.append(
            f"The article exceeded {max_chars:,} characters and was truncated for this prototype."
        )
    if len(tokenize_words(text)) < 80:
        raise ValueError(
            "Please provide at least about 80 words so the system has enough context to summarize responsibly."
        )
    return text, warnings_list


def sentence_split(text: str) -> list[str]:
    text = normalize_space(text.replace("\n", " "))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences: list[str] = []
    pending = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if pending:
            part = f"{pending} {part}".strip()
            pending = ""
        if len(tokenize_words(part)) < 4 and not re.search(r"\d", part):
            pending = part
            continue
        sentences.append(part)
    if pending:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {pending}".strip()
        else:
            sentences.append(pending)
    return sentences or ([text] if text else [])


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?%?", text.lower())


def _syllable_count(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def count_glossary_terms(text: str) -> int:
    total = 0
    lower = text.lower()
    for term in ECONOMIC_GLOSSARY:
        total += len(re.findall(rf"\b{re.escape(term)}\b", lower))
    return total


def readability_metrics(text: str) -> dict[str, float | int]:
    sentences = sentence_split(text)
    words = tokenize_words(text)
    word_count = max(1, len(words))
    sentence_count = max(1, len(sentences))
    syllables = sum(_syllable_count(word) for word in words)
    complex_words = sum(_syllable_count(word) >= 3 for word in words)
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllables / word_count
    reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return {
        "word_count": int(word_count),
        "sentence_count": int(sentence_count),
        "average_sentence_words": round(words_per_sentence, 2),
        "flesch_reading_ease": round(float(reading_ease), 2),
        "flesch_kincaid_grade": round(max(0.0, float(grade)), 2),
        "complex_word_percent": round(100.0 * complex_words / word_count, 2),
        "economics_term_mentions": int(count_glossary_terms(text)),
    }


def detect_terms(text: str, sentences: Sequence[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
    sentences = list(sentences or sentence_split(text))
    lower = text.lower()
    found: list[dict[str, Any]] = []
    for term in sorted(ECONOMIC_GLOSSARY, key=len, reverse=True):
        matches = list(re.finditer(rf"\b{re.escape(term)}\b", lower))
        if not matches:
            continue
        evidence = next(
            (sentence for sentence in sentences if re.search(rf"\b{re.escape(term)}\b", sentence, re.I)),
            "",
        )
        found.append(
            {
                "term": term,
                "count": len(matches),
                "plain_definition": ECONOMIC_GLOSSARY[term]["plain"],
                "why_it_matters": ECONOMIC_GLOSSARY[term]["life"],
                "evidence": evidence,
            }
        )
    found.sort(key=lambda row: (-row["count"], -len(row["term"]), row["term"]))
    return found[:limit]


def _preserve_capitalization(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def plain_language_rewrite(sentence: str, max_glossary_replacements: int = 4) -> str:
    output = normalize_space(sentence)
    for difficult, simple in PLAIN_LANGUAGE_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(difficult)}\b", re.I)
        output = pattern.sub(lambda match: _preserve_capitalization(match.group(0), simple), output)

    replacements = 0
    for term in sorted(PLAIN_TERM_REPLACEMENTS, key=len, reverse=True):
        if replacements >= max_glossary_replacements:
            break
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
        if pattern.search(output):
            simple = PLAIN_TERM_REPLACEMENTS[term]
            output = pattern.sub(
                lambda match: _preserve_capitalization(match.group(0), simple),
                output,
                count=1,
            )
            replacements += 1

    words = output.split()
    if len(words) > 24:
        split_match = re.search(r",\s+(?:while|although|but|and|which|because)\s+", output, re.I)
        if split_match:
            left = output[: split_match.start()].rstrip(" ,;")
            right = output[split_match.end() :].strip()
            connector = split_match.group(0).strip(" ,")
            output = f"{left}. {connector.capitalize()} {right}"
    output = re.sub(r"\s+([,.;:!?])", r"\1", output)
    output = output.strip()
    if output and output[-1] not in ".!?":
        output += "."
    return output


def extractive_summary(text: str, max_sentences: int = 6) -> list[str]:
    sentences = sentence_split(text)
    if len(sentences) <= max_sentences:
        return sentences
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(sentences)
    centroid = np.asarray(matrix.mean(axis=0))
    relevance = cosine_similarity(matrix, centroid).ravel()
    jargon_boost = np.array(
        [1.0 + min(0.30, 0.04 * count_glossary_terms(sentence)) for sentence in sentences]
    )
    length_penalty = np.array(
        [1.0 if 8 <= len(tokenize_words(sentence)) <= 42 else 0.82 for sentence in sentences]
    )
    position_bonus = np.array([1.12 if index < 3 else 1.0 for index in range(len(sentences))])
    base_scores = relevance * jargon_boost * length_penalty * position_bonus

    selected: list[int] = []
    candidates = set(range(len(sentences)))
    while candidates and len(selected) < max_sentences:
        best_index = None
        best_score = -float("inf")
        for index in candidates:
            redundancy = 0.0
            if selected:
                redundancy = max(
                    cosine_similarity(matrix[index], matrix[chosen]).item()
                    for chosen in selected
                )
            mmr_score = 0.74 * base_scores[index] - 0.26 * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        candidates.remove(best_index)
    return [sentences[index] for index in sorted(selected)]


def extract_key_phrases(text: str, limit: int = 12) -> list[dict[str, float]]:
    sentences = sentence_split(text)
    if not sentences:
        return []
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(sentences)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    ranked = sorted(zip(terms, scores), key=lambda pair: pair[1], reverse=True)
    output: list[dict[str, float]] = []
    chosen_terms: list[str] = []
    for phrase, score in ranked:
        if len(phrase) < 4 or phrase.isdigit():
            continue
        if any(phrase in chosen or chosen in phrase for chosen in chosen_terms):
            continue
        output.append({"phrase": phrase, "score": round(float(score), 4)})
        chosen_terms.append(phrase)
        if len(output) >= limit:
            break
    return output


class EconomicTopicDNN:
    """Small feed-forward neural network for economics topic classification."""

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
        )
        self.model = MLPClassifier(
            hidden_layer_sizes=(96, 48, 24),
            activation="relu",
            solver="adam",
            alpha=0.0008,
            learning_rate_init=0.002,
            max_iter=450,
            random_state=seed,
            early_stopping=False,
            n_iter_no_change=25,
        )
        self.is_fitted = False
        self._fit()

    def _fit(self) -> None:
        texts: list[str] = []
        labels: list[str] = []
        for topic, examples in TOPIC_EXAMPLES.items():
            texts.extend(examples)
            labels.extend([topic] * len(examples))
        features = self.vectorizer.fit_transform(texts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(features, labels)
        self.is_fitted = True

    @property
    def architecture(self) -> str:
        return "TF-IDF input -> Dense(96) -> Dense(48) -> Dense(24) -> topic probabilities"

    @property
    def training_iterations(self) -> int:
        return int(getattr(self.model, "n_iter_", 0))

    @property
    def loss_curve(self) -> list[float]:
        return [float(value) for value in getattr(self.model, "loss_curve_", [])]

    def score_sentences(self, sentences: Sequence[str]) -> dict[str, float]:
        if not sentences:
            return {topic: 0.0 for topic in TOPIC_KEYWORDS}
        if not self.is_fitted:
            self._fit()
        sample = list(sentences[:120])
        features = self.vectorizer.transform(sample)
        probabilities = self.model.predict_proba(features)
        average = probabilities.mean(axis=0)
        dnn_scores = {
            str(topic): float(score)
            for topic, score in zip(self.model.classes_, average)
        }

        joined = " ".join(sample).lower()
        keyword_raw: dict[str, float] = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            keyword_raw[topic] = float(
                sum(len(re.findall(rf"\b{re.escape(keyword)}\b", joined)) for keyword in keywords)
            )
        total_keyword = sum(keyword_raw.values()) or 1.0
        keyword_scores = {topic: value / total_keyword for topic, value in keyword_raw.items()}

        blended = {
            topic: 0.35 * dnn_scores.get(topic, 0.0) + 0.65 * keyword_scores.get(topic, 0.0)
            for topic in TOPIC_KEYWORDS
        }
        total = sum(blended.values()) or 1.0
        normalized = {topic: value / total for topic, value in blended.items()}
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def topic_evidence_sentence(topic: str, sentences: Sequence[str]) -> str:
    keywords = TOPIC_KEYWORDS.get(topic, ())
    best_sentence = ""
    best_score = -1
    for sentence in sentences:
        lower = sentence.lower()
        score = sum(lower.count(keyword) for keyword in keywords)
        score += 1 if re.search(r"\d", sentence) else 0
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence if best_score > 0 else (sentences[0] if sentences else "")


def detect_direction(text: str) -> str:
    tokens = tokenize_words(text)
    up_score = sum(token in UP_WORDS for token in tokens)
    down_score = sum(token in DOWN_WORDS for token in tokens)
    if up_score >= down_score + 2:
        return "rising or strengthening"
    if down_score >= up_score + 2:
        return "falling or weakening"
    return "mixed or unclear"


def topic_direction(topic: str, sentences: Sequence[str]) -> str:
    relevant = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in TOPIC_KEYWORDS.get(topic, ()))
    ]
    return detect_direction(" ".join(relevant or sentences[:4]))


def _find_sentences_with_cues(sentences: Sequence[str], cues: Iterable[str], limit: int) -> list[str]:
    cues_lower = tuple(cue.lower() for cue in cues)
    output: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(cue in lower for cue in cues_lower):
            output.append(sentence)
        if len(output) >= limit:
            break
    return output


def extract_numeric_evidence(sentences: Sequence[str], limit: int = 5) -> list[str]:
    evidence: list[str] = []
    for sentence in sentences:
        if re.search(
            r"(?:\$\s?\d|\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:million|billion|trillion|basis points|jobs|months|years)\b)",
            sentence,
            re.I,
        ):
            evidence.append(sentence)
        if len(evidence) >= limit:
            break
    return evidence


def affected_groups_for_topics(topics: Sequence[str]) -> list[str]:
    mapping = {
        "Inflation and prices": ["households", "fixed-income consumers", "retailers"],
        "Interest rates and credit": ["borrowers", "savers", "homebuyers", "small businesses"],
        "Jobs and wages": ["workers", "job seekers", "students entering the workforce", "employers"],
        "Growth and business activity": ["business owners", "workers", "local governments"],
        "Housing and real estate": ["renters", "homebuyers", "homeowners", "builders"],
        "Trade, currency, and supply chains": ["importers", "exporters", "travelers", "consumers of imported goods"],
        "Government policy and public finance": ["taxpayers", "benefit recipients", "public-service users"],
        "Markets and investing": ["investors", "retirement savers", "companies raising capital"],
        "Consumer spending and household finance": ["families", "retail businesses", "consumer lenders"],
    }
    output: list[str] = []
    for topic in topics:
        for group in mapping.get(topic, []):
            if group not in output:
                output.append(group)
    return output[:8]


def build_hierarchy(
    summary_sentences: Sequence[str],
    all_sentences: Sequence[str],
    topic_scores: dict[str, float],
    term_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    plain_bullets = [plain_language_rewrite(sentence) for sentence in summary_sentences]
    top_topics = list(topic_scores)[:3]
    causal = _find_sentences_with_cues(all_sentences, CAUSAL_CUES, 3)
    outlook = _find_sentences_with_cues(all_sentences, OUTLOOK_CUES, 3)
    numeric = extract_numeric_evidence(all_sentences, 5)
    uncertainty = [
        sentence
        for sentence in all_sentences
        if any(cue in set(tokenize_words(sentence)) for cue in UNCERTAINTY_CUES)
    ][:3]
    headline = plain_bullets[0] if plain_bullets else "The article did not provide enough text for a takeaway."
    return {
        "Level 1 - 15-second takeaway": headline,
        "Level 2 - Main developments": plain_bullets[:4],
        "Level 3 - Main economic themes": [
            {
                "topic": topic,
                "share": round(100.0 * topic_scores[topic], 1),
                "direction": topic_direction(topic, all_sentences),
                "evidence": topic_evidence_sentence(topic, all_sentences),
            }
            for topic in top_topics
        ],
        "Level 4 - Causes and evidence": {
            "possible_drivers_stated_in_article": causal or [
                "The article does not state a clear cause in a single sentence."
            ],
            "numeric_evidence": numeric or [
                "No prominent numeric evidence was detected; verify the article's supporting data."
            ],
            "important_terms": [row["term"] for row in term_rows[:8]],
        },
        "Level 5 - People and next chapter": {
            "groups_that_may_be_affected": affected_groups_for_topics(top_topics),
            "what_the_article_says_may_happen_next": outlook or [
                "The article does not give a clear next-step forecast."
            ],
            "uncertainties_to_keep_in_mind": uncertainty or [
                "The article gives limited explicit uncertainty language; do not treat one article as a complete forecast."
            ],
        },
    }


MECHANISM_TEMPLATES: dict[str, dict[str, str]] = {
    "Inflation and prices": {
        "rising or strengthening": "When prices rise faster than income, households lose purchasing power and often cut optional spending.",
        "falling or weakening": "Slower price growth can reduce pressure on budgets, but lower inflation does not automatically mean prices return to earlier levels.",
        "mixed or unclear": "Some prices may be easing while others remain high, so household experiences can differ by spending pattern.",
    },
    "Interest rates and credit": {
        "rising or strengthening": "Higher rates usually make new loans and variable-rate debt more expensive, while some savings yields may improve.",
        "falling or weakening": "Lower rates can gradually reduce some borrowing costs, although lenders and markets do not always pass through the full change immediately.",
        "mixed or unclear": "Different rates can move at different speeds, so mortgages, cards, business loans, and savings accounts may not react the same way.",
    },
    "Jobs and wages": {
        "rising or strengthening": "A stronger labor market can support income and bargaining power, but rapid wage gains can also add to business costs.",
        "falling or weakening": "Slower hiring or weaker wages can reduce household confidence and spending, which can feed back into business demand.",
        "mixed or unclear": "The number of jobs, hours worked, pay growth, and unemployment can send different signals at the same time.",
    },
    "Growth and business activity": {
        "rising or strengthening": "Stronger demand can support sales, investment, and hiring, while also testing production capacity and prices.",
        "falling or weakening": "Weaker activity can lead businesses to delay investment, reduce hiring, or cut inventories.",
        "mixed or unclear": "One part of the economy may expand while another contracts, making the overall story uneven.",
    },
    "Housing and real estate": {
        "rising or strengthening": "Higher housing costs can increase wealth for some owners but make renting or buying harder for others.",
        "falling or weakening": "Lower housing demand or prices can improve affordability for some buyers but weaken construction and owner equity.",
        "mixed or unclear": "Prices, rents, inventory, and mortgage rates can move in opposite directions.",
    },
    "Trade, currency, and supply chains": {
        "rising or strengthening": "Stronger trade flows or supply availability can reduce delays and support production, while currency changes affect import and export prices.",
        "falling or weakening": "Disruptions or weaker trade can create shortages, delays, and higher input costs that may reach consumers.",
        "mixed or unclear": "The effect depends on which products, countries, currencies, and transportation routes are involved.",
    },
    "Government policy and public finance": {
        "rising or strengthening": "More public spending or tax support can raise demand and services, but it may also increase borrowing or inflation pressure.",
        "falling or weakening": "Spending cuts or higher taxes can slow demand while improving the budget balance, depending on timing and design.",
        "mixed or unclear": "Policy creates winners, losers, delayed effects, and tradeoffs that a headline may not fully show.",
    },
    "Markets and investing": {
        "rising or strengthening": "Higher asset prices can support confidence and wealth, but they can also reflect optimistic assumptions that may change.",
        "falling or weakening": "Market declines can reduce portfolio values and make companies more cautious, even when the real economy is not yet in recession.",
        "mixed or unclear": "Stocks, bonds, currencies, and commodities often respond differently to the same news.",
    },
    "Consumer spending and household finance": {
        "rising or strengthening": "Stronger spending supports business revenue, but debt-funded spending can become difficult to sustain.",
        "falling or weakening": "Weaker spending can protect household cash flow but reduce sales for businesses and slow the economy.",
        "mixed or unclear": "Higher-income and lower-income households can react very differently to the same prices and interest rates.",
    },
}


def build_story(
    hierarchy: dict[str, Any],
    topic_scores: dict[str, float],
    all_sentences: Sequence[str],
) -> dict[str, str]:
    top_topic = next(iter(topic_scores))
    direction = topic_direction(top_topic, all_sentences)
    themes = hierarchy["Level 3 - Main economic themes"]
    supporting = themes[1]["topic"] if len(themes) > 1 else "household finances"
    next_lines = hierarchy["Level 5 - People and next chapter"][
        "what_the_article_says_may_happen_next"
    ]
    return {
        "The setting": hierarchy["Level 1 - 15-second takeaway"],
        "The change": (
            f"The article is mainly about {top_topic.lower()}, with {supporting.lower()} as another important part of the story. "
            f"The detected direction is {direction}."
        ),
        "The chain reaction": MECHANISM_TEMPLATES[top_topic][direction],
        "The people in the story": (
            "The likely affected groups include "
            + ", ".join(
                hierarchy["Level 5 - People and next chapter"][
                    "groups_that_may_be_affected"
                ][:5]
            )
            + "."
        ),
        "The next chapter": plain_language_rewrite(next_lines[0]) if next_lines else "The next step is uncertain.",
        "The caution": (
            "This is a grounded explanation of the article, not a prediction guarantee. "
            "Check the cited evidence, the original source, and newer data before making a financial decision."
        ),
    }


def _impact_record(
    area: str,
    impact: str,
    why: str,
    action: str,
    evidence: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "area": area,
        "possible_impact": impact,
        "why": why,
        "practical_check": action,
        "article_evidence": evidence,
        "confidence": round(float(max(0.05, min(0.95, confidence))), 2),
    }


def personal_impact_analysis(
    profile: ReaderProfile,
    topic_scores: dict[str, float],
    sentences: Sequence[str],
) -> list[dict[str, Any]]:
    impacts: list[dict[str, Any]] = []
    top_topics = list(topic_scores)[:5]
    for topic in top_topics:
        score = topic_scores[topic]
        if score < 0.06:
            continue
        direction = topic_direction(topic, sentences)
        evidence = topic_evidence_sentence(topic, sentences)
        confidence = 0.48 + min(0.35, score) + (0.08 if re.search(r"\d", evidence) else 0.0)

        if topic == "Inflation and prices":
            impact = (
                "Everyday expenses may take a larger share of income."
                if direction == "rising or strengthening"
                else "Price pressure may ease, but many price levels can remain above earlier levels."
            )
            impacts.append(
                _impact_record(
                    "Everyday budget",
                    impact,
                    "Inflation changes purchasing power, especially for people with a sensitive monthly budget.",
                    "Compare your own grocery, transportation, insurance, and utility costs with your income growth rather than relying only on the headline rate.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Interest rates and credit":
            debt_note = (
                "Your profile reports debt exposure, so variable-rate balances and new loans deserve extra attention."
                if "no debt" not in profile.debt_exposure.lower()
                else "Your profile reports little debt, so savings rates may matter more than borrowing rates."
            )
            impacts.append(
                _impact_record(
                    "Borrowing and saving",
                    (
                        "New borrowing may become more expensive while some savings yields may improve."
                        if direction == "rising or strengthening"
                        else "Some borrowing costs may gradually fall, while savings yields may also decline."
                    ),
                    debt_note,
                    "Check the actual annual percentage rate, fees, fixed-versus-variable terms, and after-tax savings yield before changing a loan or account.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Jobs and wages":
            role_note = (
                "As a student, the most direct channels may be internships, entry-level hiring, starting salaries, and part-time work."
                if "student" in profile.role.lower()
                else "The most direct channels may be job security, hours, promotion opportunities, and wage growth."
            )
            impacts.append(
                _impact_record(
                    "Work and income",
                    (
                        "Job opportunities and bargaining power may improve."
                        if direction == "rising or strengthening"
                        else "Hiring may become more selective and income growth may slow."
                    ),
                    role_note,
                    "Track conditions in your own field and location; national averages can hide large differences by industry and experience level.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Housing and real estate":
            if "renter" in profile.housing_status.lower():
                impact = "Rent renewals, vacancy rates, and local construction are more directly relevant than national home-price headlines."
                action = "Compare local asking rents, renewal terms, utility costs, and the cost of moving before making a housing decision."
            elif "buyer" in profile.housing_status.lower():
                impact = "The combination of home prices, mortgage rates, taxes, and insurance determines affordability."
                action = "Stress-test the full monthly payment rather than focusing only on the home price or mortgage rate."
            else:
                impact = "Home value, taxes, insurance, maintenance, and any adjustable-rate debt may be affected differently."
                action = "Separate short-term market news from your actual mortgage terms and long-term housing needs."
            impacts.append(
                _impact_record(
                    "Housing",
                    impact,
                    f"Your selected housing status is {profile.housing_status.lower()}.",
                    action,
                    evidence,
                    confidence,
                )
            )
        elif topic == "Trade, currency, and supply chains":
            impacts.append(
                _impact_record(
                    "Imported goods and travel",
                    "Prices or availability may change for imported food, fuel, electronics, vehicles, clothing, or travel.",
                    "Trade barriers, shipping delays, and exchange-rate moves can change costs before a product reaches the consumer.",
                    "Look for the specific products and countries named in the article; a broad trade headline does not affect every purchase equally.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Government policy and public finance":
            impacts.append(
                _impact_record(
                    "Taxes, benefits, and public services",
                    "The effect may appear through taxes, credits, eligibility rules, tuition support, transportation, health services, or local fees.",
                    "Fiscal-policy effects depend on who qualifies, when the policy starts, and how it is funded.",
                    "Verify official eligibility and effective dates before assuming a proposed policy changes your personal finances.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Markets and investing":
            impacts.append(
                _impact_record(
                    "Savings and investments",
                    "Market values and expected returns may change, but a one-day reaction does not establish a long-term trend.",
                    "Markets react to both the new information and how it differs from prior expectations.",
                    "Avoid making a major portfolio change from one article; compare the news with your time horizon, diversification, and risk capacity.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Growth and business activity":
            impacts.append(
                _impact_record(
                    "Local opportunities",
                    (
                        "Stronger activity may support hiring, hours, business revenue, and tax receipts."
                        if direction == "rising or strengthening"
                        else "Weaker activity may reduce hiring, hours, business investment, and local demand."
                    ),
                    "Economic growth reaches households through jobs, wages, business conditions, and public revenue.",
                    "Check your own industry's orders, hiring plans, and local conditions rather than treating national growth as uniform.",
                    evidence,
                    confidence,
                )
            )
        elif topic == "Consumer spending and household finance":
            impacts.append(
                _impact_record(
                    "Household cash flow",
                    "Changes in income, debt payments, confidence, and spending can reinforce one another.",
                    f"Your budget sensitivity is listed as {profile.budget_sensitivity.lower()}.",
                    "Review fixed expenses, emergency savings, and high-cost debt before increasing optional spending based on an upbeat headline.",
                    evidence,
                    confidence,
                )
            )

    interest_text = " ".join(profile.interests).lower()
    for impact in impacts:
        if any(word in (impact["area"] + " " + impact["possible_impact"]).lower() for word in interest_text.split()):
            impact["confidence"] = round(min(0.95, impact["confidence"] + 0.03), 2)
    return impacts[:7]


class NGramWordForecaster:
    """Transparent statistical next-word model with backoff."""

    def __init__(self, order: int = 4, content_only: bool = True) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        self.order = order
        self.content_only = content_only
        self.counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            context_length: defaultdict(Counter)
            for context_length in range(order)
        }
        self.is_fitted = False

    def fit(self, text: str) -> "NGramWordForecaster":
        words = tokenize_words(text)
        if len(words) < self.order:
            raise ValueError("The text is too short to train the word forecaster.")
        for index, target in enumerate(words):
            for context_length in range(self.order):
                if index < context_length:
                    continue
                context = tuple(words[index - context_length : index]) if context_length else ()
                self.counts[context_length][context][target] += 1
        self.is_fitted = True
        return self

    def predict(self, prefix: str, top_k: int = 5) -> list[dict[str, float | int | str]]:
        if not self.is_fitted:
            raise RuntimeError("Call fit before predict.")
        prefix_words = tokenize_words(prefix)
        for context_length in range(min(self.order - 1, len(prefix_words)), -1, -1):
            context = tuple(prefix_words[-context_length:]) if context_length else ()
            counter = self.counts[context_length].get(context)
            if not counter:
                continue
            items = [
                (word, count)
                for word, count in counter.items()
                if not self.content_only or word not in COMMON_STOPWORDS
            ]
            if not items:
                items = list(counter.items())
            total = sum(count for _, count in items) or 1
            ranked = sorted(items, key=lambda pair: (-pair[1], pair[0]))[:top_k]
            return [
                {
                    "word": word,
                    "probability": round(count / total, 4),
                    "count": int(count),
                    "context_used": " ".join(context) if context else "unigram backoff",
                }
                for word, count in ranked
            ]
        return []

    def autocomplete(self, prefix: str, additional_words: int = 8) -> str:
        generated = tokenize_words(prefix)
        if not generated:
            return prefix
        for _ in range(additional_words):
            predictions = self.predict(" ".join(generated), top_k=6)
            if not predictions:
                break
            candidates = [str(row["word"]) for row in predictions]
            next_word = next(
                (word for word in candidates if word not in generated[-3:]),
                candidates[0],
            )
            generated.append(next_word)
        return " ".join(generated)

    def common_prompts(self, limit: int = 6) -> list[str]:
        contexts = self.counts.get(2, {})
        ranked = sorted(
            contexts.items(),
            key=lambda item: sum(item[1].values()),
            reverse=True,
        )
        keyword_phrases = {
            " ".join(tokenize_words(keyword))
            for values in TOPIC_KEYWORDS.values()
            for keyword in values
        }
        specific_tokens = {
            "inflation", "deflation", "disinflation", "unemployment", "jobs",
            "hiring", "layoff", "wage", "salary", "mortgage", "rent",
            "credit", "loan", "borrowing", "yield", "gdp", "recession",
            "productivity", "tariff", "currency", "shipping", "shortage",
            "supply", "tax", "deficit", "debt", "stocks", "bonds",
            "earnings", "retail", "spending", "savings", "household",
            "interest", "rates", "prices", "market", "growth",
        }
        prompts: list[str] = []
        fallback: list[str] = []
        for context, _ in ranked:
            prompt = " ".join(context)
            if prompt in prompts or prompt in fallback:
                continue
            if (
                prompt in keyword_phrases
                or any(word in specific_tokens for word in context)
            ):
                prompts.append(prompt)
            elif any(word not in COMMON_STOPWORDS for word in context):
                fallback.append(prompt)
            if len(prompts) >= limit:
                break
        return (prompts + fallback)[:limit]


class FeedbackBandit:
    """Multi-armed bandit that learns preferred explanation format from ratings."""

    STYLES = ("bullet hierarchy", "story", "analogy", "question and answer")

    def __init__(self, path: str | Path | None = None, seed: int = RANDOM_SEED) -> None:
        self.path = Path(path) if path else None
        self.random = random.Random(seed)
        self.values = {style: 0.5 for style in self.STYLES}
        self.counts = {style: 0 for style in self.STYLES}
        self.history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.values.update({key: float(value) for key, value in payload.get("values", {}).items() if key in self.values})
            self.counts.update({key: int(value) for key, value in payload.get("counts", {}).items() if key in self.counts})
            self.history = list(payload.get("history", []))[-500:]
        except Exception:
            pass

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "values": self.values,
                    "counts": self.counts,
                    "history": self.history[-500:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def recommend(self, epsilon: float = 0.05) -> str:
        if self.random.random() < epsilon:
            return self.random.choice(self.STYLES)
        return max(self.STYLES, key=lambda style: (self.values[style], -self.counts[style]))

    def update(
        self,
        style: str,
        overall_rating: int,
        clarity_rating: int,
        usefulness_rating: int,
        comment: str = "",
    ) -> dict[str, Any]:
        if style not in self.STYLES:
            raise ValueError(f"Unknown style. Choose one of: {', '.join(self.STYLES)}")
        ratings = [overall_rating, clarity_rating, usefulness_rating]
        if any(rating < 1 or rating > 5 for rating in ratings):
            raise ValueError("Ratings must be integers from 1 to 5.")
        reward = sum((rating - 1) / 4 for rating in ratings) / len(ratings)
        self.counts[style] += 1
        count = self.counts[style]
        self.values[style] += (reward - self.values[style]) / count
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "style": style,
            "overall_rating": int(overall_rating),
            "clarity_rating": int(clarity_rating),
            "usefulness_rating": int(usefulness_rating),
            "reward": round(float(reward), 4),
            "comment": comment.strip(),
        }
        self.history.append(record)
        self._save()
        return record

    def status(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "style": style,
                    "estimated_reward": round(self.values[style], 4),
                    "feedback_count": self.counts[style],
                }
                for style in self.STYLES
            ]
        ).sort_values(["estimated_reward", "feedback_count"], ascending=[False, False])


class OptionalTransformerExplainer:
    """Lazy, optional FLAN-T5 explainer. The deterministic pipeline remains the source of evidence."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "The optional neural component requires transformers, sentencepiece, and torch. "
                "Run the installation cell first."
            ) from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        self.load()
        import torch

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    @staticmethod
    def unsupported_numbers(generated: str, source: str) -> list[str]:
        pattern = r"(?<![A-Za-z])\$?\d+(?:[.,]\d+)*(?:%|\s?(?:million|billion|trillion|basis points))?"
        source_numbers = {value.replace(",", "") for value in re.findall(pattern, source, re.I)}
        generated_numbers = {value.replace(",", "") for value in re.findall(pattern, generated, re.I)}
        return sorted(generated_numbers - source_numbers)

    def enhance(
        self,
        source_notes: Sequence[str],
        terms: Sequence[dict[str, Any]],
        profile: ReaderProfile,
    ) -> dict[str, Any]:
        evidence_block = " ".join(source_notes)[:3000]
        term_block = "; ".join(
            f"{row['term']}: {row['plain_definition']}" for row in terms[:8]
        )
        summary_prompt = (
            "Rewrite the source notes as five short plain-English bullets for a high-school reader. "
            "Keep all claims grounded in the notes. Do not add names, numbers, dates, forecasts, or causes that are not present. "
            f"Source notes: {evidence_block} Terminology help: {term_block}"
        )
        story_prompt = (
            "Turn the source notes into a concise economics story with these labels: Setting, Change, Chain reaction, Everyday impact, What to watch. "
            "Use cautious language. Do not add facts. The reader profile is: "
            f"country={profile.country}, role={profile.role}, housing={profile.housing_status}, debt={profile.debt_exposure}. "
            f"Source notes: {evidence_block}"
        )
        neural_summary = self._generate(summary_prompt, max_new_tokens=190)
        neural_story = self._generate(story_prompt, max_new_tokens=240)
        unsupported = sorted(
            set(
                self.unsupported_numbers(neural_summary, evidence_block)
                + self.unsupported_numbers(neural_story, evidence_block)
            )
        )
        return {
            "model": self.model_name,
            "device": self.device,
            "plain_summary": neural_summary,
            "story": neural_story,
            "unsupported_numbers_detected": unsupported,
            "grounding_note": (
                "The neural text is an optional rewrite. The extractive evidence and original article remain the source of truth."
            ),
        }


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        return True


def fetch_public_article(url: str, timeout_seconds: int = 20) -> dict[str, str]:
    url = url.strip()
    if not _is_public_http_url(url):
        raise ValueError("Enter a public http or https URL. Local and private-network addresses are blocked.")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; EconoLensEducationalPrototype/1.0; "
            "+https://streamlit.io/)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if not any(kind in content_type for kind in ("text", "html", "xml", "json")):
        raise ValueError(f"The URL returned unsupported content type: {content_type or 'unknown'}")
    if len(response.content) > 6_000_000:
        raise ValueError("The webpage is too large for this educational prototype.")
    html = response.text
    title = "Untitled webpage"
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        title = normalize_space(soup.title.string)

    extracted = ""
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            ) or ""
        except Exception:
            extracted = ""
    if not extracted:
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        extracted = soup.get_text(" ")
    extracted = normalize_space(extracted)
    if len(tokenize_words(extracted)) < 80:
        raise ValueError(
            "The article text could not be extracted. Paste the article text directly. This tool does not bypass paywalls or access controls."
        )
    return {"title": title, "source": urlparse(url).netloc, "url": url, "text": extracted}


def compare_readability(original: str, simplified: str) -> pd.DataFrame:
    original_metrics = readability_metrics(original)
    simple_metrics = readability_metrics(simplified)
    rows = []
    for metric in (
        "word_count",
        "average_sentence_words",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "complex_word_percent",
        "economics_term_mentions",
    ):
        rows.append(
            {
                "metric": metric,
                "original": original_metrics[metric],
                "simplified": simple_metrics[metric],
                "change": round(float(simple_metrics[metric]) - float(original_metrics[metric]), 2),
            }
        )
    return pd.DataFrame(rows)


def _quality_flags(
    article: str,
    topic_scores: dict[str, float],
    numeric_evidence: Sequence[str],
    term_rows: Sequence[dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    word_count = len(tokenize_words(article))
    if word_count < 250:
        flags.append("The input is short, so topic and cause detection may be less stable.")
    if not numeric_evidence:
        flags.append("Few or no numeric evidence sentences were detected; check the source for supporting data.")
    if topic_scores and max(topic_scores.values()) < 0.24:
        flags.append("No single economic theme dominates; treat the topic ranking as mixed.")
    if len(term_rows) >= 15:
        flags.append("The article contains many economics terms; review the glossary before relying on the simplified story.")
    flags.append("Personal impacts are plausible pathways, not individualized financial advice or guaranteed outcomes.")
    return flags


def make_markdown_report(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    hierarchy = result["hierarchy"]
    lines = [
        f"# {APP_NAME} Analysis",
        "",
        f"**Article:** {metadata.get('title', 'Untitled economics article')}",
        f"**Source:** {metadata.get('source', 'User-provided text')}",
        f"**Author:** {STUDENT_NAME}",
        f"**Advisor:** {ADVISOR_NAME}",
        f"**Analysis ID:** `{result['analysis_id']}`",
        "",
        "## Level 1 - 15-second takeaway",
        hierarchy["Level 1 - 15-second takeaway"],
        "",
        "## Level 2 - Main developments",
    ]
    lines.extend(f"- {item}" for item in hierarchy["Level 2 - Main developments"])
    lines.extend(["", "## Level 3 - Main economic themes"])
    for theme in hierarchy["Level 3 - Main economic themes"]:
        lines.append(
            f"- **{theme['topic']}** ({theme['share']}%, {theme['direction']}): {theme['evidence']}"
        )
    lines.extend(["", "## The story"])
    for label, paragraph in result["story"].items():
        lines.append(f"### {label}")
        lines.append(paragraph)
    lines.extend(["", "## Possible personal impacts"])
    for impact in result["personal_impacts"]:
        lines.append(
            f"### {impact['area']} (confidence {impact['confidence']:.0%})\n"
            f"**Possible impact:** {impact['possible_impact']}\n\n"
            f"**Why:** {impact['why']}\n\n"
            f"**Practical check:** {impact['practical_check']}\n\n"
            f"**Article evidence:** {impact['article_evidence']}"
        )
    lines.extend(["", "## Important terms"])
    for row in result["terms"][:10]:
        lines.append(
            f"- **{row['term']}**: {row['plain_definition']} {row['why_it_matters']}"
        )
    lines.extend(["", "## Evidence sentences"])
    lines.extend(f"- {sentence}" for sentence in result["source_summary"])
    lines.extend(["", "## Quality and safety notes"])
    lines.extend(f"- {flag}" for flag in result["quality_flags"])
    lines.append("")
    lines.append(f"*Author: {STUDENT_NAME} | Advisor: {ADVISOR_NAME}*")
    lines.append("")
    lines.append("*Educational prototype. Not financial, investment, legal, or tax advice.*")
    return "\n".join(lines)


class EconomicsNewsPipeline:
    def __init__(self, feedback_path: str | Path | None = None) -> None:
        self.topic_model = EconomicTopicDNN()
        self.feedback = FeedbackBandit(feedback_path)
        self.word_forecaster = NGramWordForecaster(order=4, content_only=True)
        self.neural_explainer: OptionalTransformerExplainer | None = None

    def analyze(
        self,
        article_text: str,
        profile: ReaderProfile | None = None,
        metadata: ArticleMetadata | None = None,
        use_neural: bool = False,
    ) -> dict[str, Any]:
        profile = profile or ReaderProfile()
        metadata = metadata or ArticleMetadata()
        article, input_warnings = clean_article_text(article_text)
        sentences = sentence_split(article)
        source_summary = extractive_summary(article, max_sentences=7)
        terms = detect_terms(article, sentences=sentences, limit=24)
        topic_scores = self.topic_model.score_sentences(sentences)
        hierarchy = build_hierarchy(source_summary, sentences, topic_scores, terms)
        story = build_story(hierarchy, topic_scores, sentences)
        personal_impacts = personal_impact_analysis(profile, topic_scores, sentences)
        key_phrases = extract_key_phrases(article, limit=14)
        plain_summary = hierarchy["Level 2 - Main developments"]
        simplified_text = " ".join(plain_summary)
        readability_table = compare_readability(article, simplified_text)
        self.word_forecaster = NGramWordForecaster(order=4, content_only=True).fit(article)
        forecast_examples: dict[str, list[dict[str, Any]]] = {}
        for prompt in self.word_forecaster.common_prompts(limit=5):
            forecast_examples[prompt] = self.word_forecaster.predict(prompt, top_k=5)

        numeric_evidence = hierarchy["Level 4 - Causes and evidence"]["numeric_evidence"]
        real_numeric_evidence = [
            item for item in numeric_evidence if not item.startswith("No prominent")
        ]
        quality_flags = input_warnings + _quality_flags(
            article, topic_scores, real_numeric_evidence, terms
        )
        result: dict[str, Any] = {
            "app": APP_NAME,
            "tagline": APP_TAGLINE,
            "student": STUDENT_NAME,
            "advisor": ADVISOR_NAME,
            "analysis_id": hashlib.sha256(article.encode("utf-8")).hexdigest()[:12],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": asdict(metadata),
            "profile": asdict(profile),
            "source_text": article,
            "source_summary": source_summary,
            "plain_summary": plain_summary,
            "hierarchy": hierarchy,
            "story": story,
            "topic_scores": {topic: round(float(score), 6) for topic, score in topic_scores.items()},
            "topic_directions": {
                topic: topic_direction(topic, sentences) for topic in list(topic_scores)[:5]
            },
            "terms": terms,
            "key_phrases": key_phrases,
            "personal_impacts": personal_impacts,
            "readability": readability_table.to_dict(orient="records"),
            "forecast_examples": forecast_examples,
            "recommended_style": self.feedback.recommend(epsilon=0.0),
            "quality_flags": quality_flags,
            "neural": None,
        }

        if use_neural:
            try:
                self.neural_explainer = self.neural_explainer or OptionalTransformerExplainer()
                result["neural"] = self.neural_explainer.enhance(
                    source_summary, terms, profile
                )
                if result["neural"]["unsupported_numbers_detected"]:
                    result["quality_flags"].append(
                        "The optional neural rewrite introduced number-like text not found in the extractive source notes. Use the evidence view instead."
                    )
            except Exception as exc:
                result["neural"] = {
                    "error": str(exc),
                    "grounding_note": "The deterministic analysis completed successfully without the optional transformer.",
                }
        result["markdown_report"] = make_markdown_report(result)
        return result


def save_result_bundle(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "econolens_analysis.json"
    markdown_path = output_path / "econolens_report.md"
    impacts_path = output_path / "personal_impacts.csv"
    topics_path = output_path / "topic_scores.csv"
    terms_path = output_path / "economic_terms.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(result["markdown_report"], encoding="utf-8")
    pd.DataFrame(result["personal_impacts"]).to_csv(impacts_path, index=False)
    pd.DataFrame(
        [{"topic": topic, "score": score} for topic, score in result["topic_scores"].items()]
    ).to_csv(topics_path, index=False)
    pd.DataFrame(result["terms"]).to_csv(terms_path, index=False)
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "personal_impacts_csv": str(impacts_path),
        "topic_scores_csv": str(topics_path),
        "terms_csv": str(terms_path),
    }


def run_self_checks() -> dict[str, bool]:
    sample = (
        "Consumer prices rose 3 percent as rent and insurance remained high. "
        "The central bank held its benchmark interest rate because inflation was still above its goal. "
        "Employers added jobs, but wage growth slowed. Analysts said future rate cuts may depend on new price data. "
        "Higher borrowing costs reduced home sales and made credit card balances more expensive for some households. "
        "Retail spending continued to grow, although lower-income consumers became more cautious. "
        "Businesses reported better supply chains and shorter delivery times. "
        "The outlook remains uncertain because energy prices and global trade conditions could change. "
        "Families are comparing monthly bills with income growth and reducing optional purchases."
    )
    checks: dict[str, bool] = {}
    cleaned, _ = clean_article_text(sample)
    checks["clean_text"] = len(cleaned) > 100
    checks["sentence_split"] = len(sentence_split(sample)) >= 6
    checks["summary"] = len(extractive_summary(sample, 4)) == 4
    checks["terms"] = any(row["term"] == "inflation" for row in detect_terms(sample))
    dnn = EconomicTopicDNN()
    scores = dnn.score_sentences(sentence_split(sample))
    checks["topic_model"] = abs(sum(scores.values()) - 1.0) < 1e-6
    forecaster = NGramWordForecaster().fit(sample)
    checks["word_forecaster"] = bool(forecaster.predict("interest rate"))
    bandit = FeedbackBandit()
    record = bandit.update("story", 5, 4, 5, "clear")
    checks["feedback_bandit"] = record["reward"] > 0.8
    return checks