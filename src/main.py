from src.constants import ROOT

fp = ROOT / "resources" / "pt_words.txt"
words = []
for line in fp.read_text().split("\n"):
    words += [line.split("/")[0].split("[")[0]]

(ROOT / "temp.txt").write_text("\n".join(words))

chars = "".join(set("".join(words).lower()))
'ríêipvuõ4:ãgáüàèéóôâîçú
àáâãåāèéêëěėēîïíìįīĵłñńôöòóøōõšśûüùúūÿýžźżçćčñń