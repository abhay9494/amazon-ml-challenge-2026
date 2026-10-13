"""Administrative-region vocabularies (states, regions, departments, countries).

An address *component* (text between commas) that is exactly one of these strings is dropped by
normalize.norm_addr: the S1/S2/S3 sources disagree on whether and how the state is written
("NC" / "North Carolina", "MH" / "Maharashtra" / native script, "Nord" / "Hauts-de-France"),
and the state carries almost no information once blocking is already done within a country.
Strings are stored lower-case with non-alphanumerics replaced by single spaces.
"""
import re

US = """alabama al|alaska ak|arizona az|arkansas ar|california ca|colorado co|connecticut ct|delaware de|
florida fl|georgia ga|hawaii hi|idaho id|illinois il|indiana in|iowa ia|kansas ks|kentucky ky|louisiana la|
maine me|maryland md|massachusetts ma|michigan mi|minnesota mn|mississippi ms|missouri mo|montana mt|
nebraska ne|nevada nv|new hampshire nh|new jersey nj|new mexico nm|new york ny|north carolina nc|
north dakota nd|ohio oh|oklahoma ok|oregon or|pennsylvania pa|rhode island ri|south carolina sc|
south dakota sd|tennessee tn|texas tx|utah ut|vermont vt|virginia va|washington wa|west virginia wv|
wisconsin wi|wyoming wy|district of columbia dc|puerto rico pr|guam gu|virgin islands vi|
american samoa as|northern mariana islands mp"""

INDIA = """andhra pradesh ap|arunachal pradesh ar|assam as|bihar br|chhattisgarh cg|chattisgarh ct|goa ga|
gujarat gj|haryana hr|himachal pradesh hp|jharkhand jh|karnataka ka|kerala kl|madhya pradesh mp|
maharashtra mh|manipur mn|meghalaya ml|mizoram mz|nagaland nl|odisha od|orissa or|punjab pb|rajasthan rj|
sikkim sk|tamil nadu tn|telangana tg|telengana ts|tripura tr|uttar pradesh up|uttarakhand uk|
uttaranchal ua|west bengal wb|andaman and nicobar islands an|andaman nicobar an|chandigarh ch|
dadra and nagar haveli dn|daman and diu dd|dadra and nagar haveli and daman and diu dh|delhi dl|
nct of delhi dl|national capital territory of delhi dl|jammu and kashmir jk|jammu kashmir jk|ladakh la|
lakshadweep ld|puducherry py|pondicherry py|rajsthan rj|maharastra mh|karnatka ka|telanagana tg|
chhatisgarh cg|uttrakhand uk|orrisa or|tamilnadu tn|westbengal wb|bengal wb|keralam kl"""

FRANCE = """auvergne rhone alpes|bourgogne franche comte|bretagne|centre val de loire|corse|grand est|
hauts de france|ile de france|normandie|nouvelle aquitaine|occitanie|pays de la loire|
provence alpes cote d azur|provence alpes cote dazur|paca|guadeloupe|martinique|guyane|la reunion|reunion|
mayotte|ain|aisne|allier|alpes de haute provence|hautes alpes|alpes maritimes|ardeche|ardennes|ariege|aube|
aude|aveyron|bouches du rhone|calvados|cantal|charente|charente maritime|cher|correze|cote d or|cote dor|
cotes d armor|cotes darmor|creuse|dordogne|doubs|drome|eure|eure et loir|finistere|gard|haute garonne|gers|
gironde|herault|ille et vilaine|indre|indre et loire|isere|jura|landes|loir et cher|loire|haute loire|
loire atlantique|loiret|lot|lot et garonne|lozere|maine et loire|manche|marne|haute marne|mayenne|
meurthe et moselle|meuse|morbihan|moselle|nievre|nord|oise|orne|pas de calais|puy de dome|
pyrenees atlantiques|hautes pyrenees|pyrenees orientales|bas rhin|haut rhin|rhone|haute saone|
saone et loire|sarthe|savoie|haute savoie|seine maritime|seine et marne|yvelines|deux sevres|somme|tarn|
tarn et garonne|var|vaucluse|vendee|vienne|haute vienne|vosges|yonne|territoire de belfort|essonne|
hauts de seine|seine saint denis|val de marne|val d oise|val doise"""

# State names written in Indic scripts in S2/S3 (the only native-script address components).
NATIVE_STATES = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh", "ಕರ್ನಾಟಕ": "karnataka",
    "தமிழ்நாடு": "tamil nadu", "ગુજરાત": "gujarat", "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana",
    "हरियाणा": "haryana", "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}

COUNTRIES ="us|usa|u s a|u s|united states|united states of america|america|india|bharat|france|fr|republique francaise"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _collapse(s: str) -> str:
    return re.sub(r"([a-z])\1+", r"\1", s)


def _build():
    out = set()
    for block in (US, INDIA):
        for item in block.replace("\n", "").split("|"):
            parts = item.strip().split(" ")
            code = parts[-1]
            name = " ".join(parts[:-1])
            out.add(_norm(name))
            out.add(code)
    for block in (FRANCE, COUNTRIES):
        for item in block.replace("\n", "").split("|"):
            out.add(_norm(item))
    from unidecode import unidecode

    for native in NATIVE_STATES:
        out.add(_norm(unidecode(native)))
    out |= {_collapse(x) for x in list(out)}
    out.discard("")
    return out


REGION_STRINGS = _build()
