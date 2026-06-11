"""
confederations.py
-----------------
Maps national teams to their FIFA confederation and assigns each
confederation a relative strength multiplier.

This is used to weight the Dixon-Coles fit by strength of schedule: a
result against a team from a deep confederation (UEFA, CONMEBOL) is more
informative than the same result against a weaker-confederation side, so
matches involving weaker regions count for less.

The multipliers are deliberately coarse — they encode the broad reality
that European and South American football is, on average, stronger than
other regions — and are normalised to mean 1 inside the fit, so only the
*relative* ordering matters.
"""

# Relative strength of each confederation (higher = stronger pool).
CONFEDERATION_STRENGTH = {
    "UEFA":     1.00,   # Europe
    "CONMEBOL": 1.00,   # South America
    "CAF":      0.65,   # Africa
    "CONCACAF": 0.60,   # North/Central America & Caribbean
    "AFC":      0.50,   # Asia & Australia
    "OFC":      0.35,   # Oceania
}

# Default for any team not found below (treated as a weaker, minor side).
DEFAULT_STRENGTH = 0.55


CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}

UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Czechia", "Czechoslovakia", "Denmark", "England",
    "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany",
    "East Germany", "Gibraltar", "Greece", "Hungary", "Iceland", "Israel",
    "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Montenegro", "Netherlands",
    "North Macedonia", "Macedonia", "Northern Ireland", "Norway", "Poland",
    "Portugal", "Republic of Ireland", "Ireland", "Romania", "Russia",
    "Soviet Union", "San Marino", "Scotland", "Serbia", "Serbia and Montenegro",
    "Yugoslavia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
    "Turkey", "Ukraine", "Wales",
}

AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China PR", "China", "Chinese Taipei", "Taiwan", "Guam",
    "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
    "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives",
    "Mongolia", "Myanmar", "Nepal", "North Korea", "Korea DPR", "Oman",
    "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
    "Singapore", "South Korea", "Korea Republic", "Sri Lanka", "Syria",
    "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
    "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
}

CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Congo DR", "Djibouti", "Egypt", "Equatorial Guinea",
    "Eritrea", "Eswatini", "Swaziland", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia",
    "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius",
    "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "São Tomé and Príncipe", "Sao Tome and Principe", "Senegal", "Seychelles",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
    "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
}

CONCACAF = {
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
    "Belize", "Bermuda", "British Virgin Islands", "Canada", "Cayman Islands",
    "Costa Rica", "Cuba", "Curaçao", "Curacao", "Dominica", "Dominican Republic",
    "El Salvador", "French Guiana", "Grenada", "Guadeloupe", "Guatemala",
    "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico",
    "Montserrat", "Nicaragua", "Panama", "Puerto Rico", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Sint Maarten",
    "Suriname", "Trinidad and Tobago", "Turks and Caicos Islands",
    "United States", "US Virgin Islands",
}

OFC = {
    "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand",
    "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga",
    "Tuvalu", "Vanuatu",
}

# Build a single reverse lookup: team -> confederation name.
_TEAM_TO_CONFEDERATION: dict[str, str] = {}
for _conf, _members in (
    ("UEFA", UEFA), ("CONMEBOL", CONMEBOL), ("AFC", AFC),
    ("CAF", CAF), ("CONCACAF", CONCACAF), ("OFC", OFC),
):
    for _team in _members:
        _TEAM_TO_CONFEDERATION[_team] = _conf


def confederation(team: str) -> str | None:
    """Return the confederation name for a team, or None if unknown."""
    return _TEAM_TO_CONFEDERATION.get(team)


def confederation_strength(team: str) -> float:
    """
    Return the strength multiplier for a team's confederation.
    Unknown teams fall back to DEFAULT_STRENGTH.
    """
    conf = _TEAM_TO_CONFEDERATION.get(team)
    if conf is None:
        return DEFAULT_STRENGTH
    return CONFEDERATION_STRENGTH[conf]
