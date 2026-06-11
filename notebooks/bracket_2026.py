"""
bracket_2026.py
---------------
The official 2026 World Cup group draw (48 teams, 12 groups A-L), with
team names mapped to the spellings used in results.csv.

Single source of truth shared by the prediction scripts.

Name mappings applied:
    Korea Republic -> South Korea     Côte d'Ivoire -> Ivory Coast
    Czechia        -> Czech Republic  Cabo Verde    -> Cape Verde
    Congo          -> DR Congo        Türkiye       -> Turkey
    USA            -> United States   IR Iran       -> Iran

(Group K "Congo" is the DR Congo playoff winner, not Rep. of the Congo.)
"""

BRACKET_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Turkey", "Australia"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
