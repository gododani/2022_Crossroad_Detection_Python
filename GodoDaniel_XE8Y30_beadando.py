import cv2
import numpy as np


def add_noise(img_in, percentage, value):
    noise = np.copy(img_in)
    # kép magasság * kép szélesség * százalék
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)

    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])

        # kétcsatornás kép
        if img_in.ndim == 2:
            noise[j, i] = value

        # háromcsatornás kép
        if img_in.ndim == 3:
            noise[j, i] = [value, value, value]
    return noise


def add_salt_and_pepper(img_in, percentage1, percentage2):
    # Só hozzáadása
    salt = add_noise(img_in, percentage1, 255)
    # Bors hozzáadása
    pepper = add_noise(salt, percentage2, 0)
    return pepper


# kép beolvasás
img = cv2.imread('crosswalk.jpg', cv2.IMREAD_COLOR)
#img = cv2.imread('crosswalk2.jpg', cv2.IMREAD_COLOR)
img = add_salt_and_pepper(img, 0.01, 0.01)
cv2.imshow('Default image', img)

# ----- Morfológiai szűrés -----
# Strukturáló elem definiálása téglalap formával és 3x3-es mérettel
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilatáció -> erózió -> erózió -> diláció az eredeti képen strukturáló elemmel és kép megjelenítés
segment_filtered = cv2.dilate(img, struct)
segment_filtered = cv2.erode(segment_filtered, struct, iterations=2)
segment_filtered = cv2.dilate(segment_filtered, struct)
cv2.imshow('3x3 morphology filter', segment_filtered)

# ----- Gaus szűrés -----
# Paraméterek: (1) forráskép,(2) maszk mérete (páratlan), (3) X-irányú szórás, (4) Y-irányú szórás
gaussSmoothing = cv2.GaussianBlur(segment_filtered, (3, 3), sigmaX=2.0, sigmaY=2.0)
cv2.imshow('Gauss Smoothing', gaussSmoothing)

# Globlis küszöbölés használata
lower = (75, 75, 75)
upper = (180, 180, 180)
threshold = cv2.inRange(gaussSmoothing, lower, upper)
cv2.imshow("Threshold", threshold)

# Kontúrok
# ---- Paraméterek ----
# thresh: Bemeneti kép
# cv2.RETR_EXTERNAL: Csak a legkülső objektumok külső kontúrjait kapjuk
# cv2.CHAIN_APPROX_SIMPLE: Csak a sarokpontokat menti el a kontúrból
# ---- EREDMÉNYEK ----
# cntrs: Kontúrokból álló lista
# hierarchy: A kontúrok topológiai rendszere
cntrs, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = img.copy()
for c in cntrs:
    # elmenti az adott kontúr pontjainakszámát
    area = cv2.contourArea(c)
    if area > 850:
        # ---- Paraméterek ----
        # contours: Kép amire rajzolunk
        # [c]: Kontúrok tömbje
        # -1: Összes kontúrt rajzoljuk
        # (0, 0, 255): Piros szín
        # cv2.FILLED / -1: Kitölti a belső területeket is
        cv2.drawContours(contours, [c], -1, (0, 0, 255), cv2.FILLED)
cv2.imshow("Contours", contours)
cv2.imwrite("Result.png", contours)

cv2.waitKey(0)
cv2.destroyAllWindows()

# R5: Felfestési hiányosságok, árnyék hatása, gyalogos/autó takarás esetén megoldási lehetőségek.
# -----------------------------------------------------------------------------------------------
# Egy probléma:
# A feladatmegoldás a 2 mintaképre működik, de más képen szinte biztos hogy nem, mert a használt globális küszöbölési
# értékek például egy olyan képen ahol az aszfalt világosabb, mint az itt felhasznált alsó küszöb, akkor
# eredményül az eredmény képen az aszfalt is piros lesz.

# Egy lehetséges megoldás:
# Dinamikusan egy algoritmus által meghatározni a felső és alsó küszöböt a képen.

# Egy probléma: Ha például egy kocsi levágja a szélét a zebrának, akkor nem biztos, hogy a levágott rész hossza megfelel
# majd a kontúrvizsgálati feltételnek. Ezt lehetetlen dinamikusan meghatározni, mert lehet olyan részt is
# elfogadna ami nem a zebrához tarozik, hanem a kép egy másik részéhez. Ez azért fordulhatna elő, mert benne lenne az
# alsó és felső küszöb tartományban, mint például a 2 mintaképnek a zebra alatti meg feletti vöröses része.

# Egy lehetséges megoldás:
# Forma detektáció használata, párhuzamosság, merőlegesség és távolság vizsgálata. Ezekkel meg lehet határozni, hogy egy
# küszöbtartományon belüli érték például merőleges-e zebra széleit összekötő vonalra, illetve párhuzamos-e a többi
# zebrával
