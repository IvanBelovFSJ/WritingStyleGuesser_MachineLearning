# -*- coding: utf-8 -*-
"""
Created on Tue May 21 01:56:03 2019

@author: Ivan Belov
"""
from sklearn import tree
# 1st Variable -- n. of pages,
# 2nd Variable -- glossy - 1 / matte - 0,
# 3rd Variable -- hard - 0 / soft cover - 1,
# 4th Variable -- n. of cnapters,
# 5th Variable -- n. of ads., or promotions,
# 6th Variable -- n. of authors,
# 7th Variable -- year published.
features = [
        [524, 0,0, 9, 0 , 1, 2001 ], # 2001 - S. Dovlatov, Zapavednik, ISBN 5-267-00506-1
            [243, 1, 1, 2, 8, 1, 2007 ], # 2007 - S. Dovlatov, Solo na Underwood, Solo na IBM, ISBN 978-5-91181-382-6
            [154, 1, 1,7, 1, 1, 2006 ], # 2006 - # S. Dovlatov, Gizn' korotka, ISBN 5-352-00511-9
            [155, 1, 1, 9, 2, 1, 2008 ], # 2008 - S. Dovlatv,  Chemodan, ISBN 978-5-91181-427-4
            [155, 1, 1, 12, 2, 1, 2007], # 2007 - # S. Dovlatov, Inostranka, ISBN 978-5-91181-585-1
            [189, 1, 1, 2, 0, 1, 2007] , # 2007 - S. Dovlatov, Remeslo, ISBN 5-91181-202-9
            [223, 1, 1, 20, 0, 1, 2013] , # 2013 - Smithsonian Nature Guide : GEMS, ISBN 978-1-4654-0218-9
            [352, 1, 1, 31, 0, 1, 2012 ], # 2012 - Smithsonian Nature Guide : ROCKS AND MINERALS, ISBN 978-0-7566-9042-7
            [360, 1, 1, 49, 0, 3, 2005 ], # 2005 - Smithsonian : ROCK AND GEM, ISBN 978-0-7566-3342-4
            [261, 1, 1, 16, 0, 2, 1990], # 2003 - Rich is a state of mind,  ISBN 978-0-9731849-0-7
            [243, 1, 1, 7, 0, 1, 1992 ], # 1992 - J.S. Bolen, Ring of Power, ISBN 0-06-251001-0
            [312, 0, 0, 12, 0, 1, 2001 ], # 2001 - P.C. McGraw, Self Matters, ISBN 0-7432-2423-x
            [209, 1, 1, 9, 0, 1, 2006 ], # 2006 - Richard Moore, Escaping the Matrix, ISBN 0-9770983-0-3
            [608, 0, 0, 6, 1, 1, 1995 ], # 1995 - G. Edward Griffin, The Creature from Jekyll Island, ISBN 0-912986-21-2
            [184, 0, 0, 16, 0, 1, 1983 ], # 1983 - Peter Newman, True North Not Strong And Free, ISBN 0-7710-6798-4
            [275, 0, 0, 21, 0, 1, 1966], # M. Shulman, Anyone Can Make a Million, Library of Congress Catalog Card Number 66-27631
            [197, 0, 0, 10, 1, 1, 1989], # David Chilton, The Wealthy Barber, ISBN 0-9683947-3-6
            [392, 1, 1, 7, 4, 2, 1979], # After The Cataclysm, Post War Indochina & The Reconstruction of Imperial Ideology, N. Chomsky, Edward S. Herman, ISBN 0-919618-90-1
            [434, 1, 1, 5, 0, 2, 1979], # The Washington Connection and third World Fascism , N. Chomsky, Edward S. Herman, ISBN 0-919618-90-1
            [404, 1, 1, 10, 1, 1, 1967], # American Power And The New Mandarines, N. Chomsky
            [227, 1, 1, 10, 0, 2, 2017], # Global Discontents, N. Chomsky, ISBN 978-1-250-14618-2,
            [206, 1, 1, 10, 2, 1, 1995], # Pirates and Emperors, N. Chomsky, ISBN 1-894531-20-4
            [252, 1, 1, 14, 1, 1, 2000], # Rogue States, The Rule of Force in Wolrd Affairs, N. Chomsky, ISBN 0-89608-611-9
            [101, 1, 1, 14, 1, 1, 2002], # Media Control 2ND Edition, N. Chomsky, ISBN 1-58322-536-6
            [140, 1, 1, 12, 1, 1, 2002], # 9-11, N. Chomsky, ISBN 1-58322-489-0
            [158, 1, 1, 14, 0, 3, 2003], # Power and Terror post 9-11 talks and interviews, N. Chomsky, ISBN 1-58322-590-0
            [167, 1, 1, 19, 0, 1, 1993], # Letts From Lexington, Reflictions on propaganda, N. Chomsky, ISBN 0-921284-77-2
            [231, 1, 1, 7, 0, 1, 2002], # Pirates and Emperors, Old and New, International Terrorism in the real world, N. Chomsky, ISBN 1-896357-63-6,
            [95, 0, 0, 2, 0, 1, 1972], # Problems of Knowledge and Freedom, N. Chomsky, ISBN 0-214653-71-4
            [311, 0, 0, 8, 0, 1, 1994], # World Orders Old and New ,N. Chomsky, ISBN 0-231101-56-2
            [404, 1, 1,4, 0, 2, 1987], # The Chomsky Reader ,N. Chomsky, ISBN 0-394-75173-6
            [274, 0, 0,9, 0, 1, 2003], # Hegemony of Survival, America's quest for Global Dominance ,N. Chomsky, ISBN 0-8050-7400-7
            [276, 1, 1, 3, 0, 1, 2005], # Everything is Illuminated , J. Safran F. , ISBN 0-06-079217-5
            [480, 1, 1, 4, 0, 1, 1995], # Dear Theo, Irwing Stone , ISBN 0-452-27804-0
            [420, 1, 1, 13, 0, 10, 2006], # Northern California, Fodor's , ISBN 1-4000-1602-9
            [96, 0, 0, 12, 0, 1, 1965], # The Story of Canadian Flag, Stanley , ISBN 555
            [280, 1, 1, 5, 0, 1, 2001], # Klondike Tales, Jack London , ISBN 978-0-375-75685-6
            [837, 1, 0, 11, 0, 1, 2017], # Mathematics for Computing Science , ISBN 978-0-17-676509-5
            [893, 1, 1, 20, 0, 3, 1998], # COBOL From Micro to Mainframe , ISBN 0-13-790817-2
            [1049, 1, 1, 24, 0, 1, 2007], # Managing and troubleshooting PCs , ISBN 978-0-07-226356-5
            [993, 1, 0, 30, 0, 1, 2005], # Basic Technicals Mathematics with Calculus , ISBN 0-321-30689-9
            [1084, 1, 0, 24, 0, 1, 2003], # Chemistry, The Molecular Nature of Matter and Change, ISBN 0-07-119894-6
            [448, 1, 1, 13, 0, 1, 2013], # Becoming an Active Reader, ISBN 978-0-19-901906-9
            [382, 0, 0, 45, 0, 1, 2018], # Past tense, Lee Child, ISBN 978-0-39-959351-2
            [588, 1, 1, 15, 0, 1, 2013], # Java for everyone, Cay Horstmann, ISBN 978-1-118-06331-6
            [251, 1, 1, 3, 0, 1, 1923], # Dog's Heart, M. Bulghakov, ISBN 978-5-389-01364-3
            [288, 1, 1, 16, 0, 9, 2012], # Van Gogh, Up Close, ISBN 978-0-88884-895-6
            [320, 0, 0, 8, 0, 1, 2006], # Henry Wadworth Longfellow, ISBN 5-94320-039-8
            [673, 1, 1, 0, 0, 1, 1981], # Collins Gem, Latin-English English-Latin DICSTIONARY ISBN 0-00-458644-1
            [431, 0, 1, 3, 0, 1, 2000], # Andrey Makarevich , Seven Thousand Cities ISBN 5-04-003999-9
            [317, 1, 1, 11, 0, 1, 2013], # Yiddish , Dictionary & Phrasebook ISBN 978-0-7818-1298-6
            [991, 0, 1, 0, 0, 1, 2003], # Sirotina, English-Russian Dictionary ISBN 589-886-010-X
            [598, 0, 1, 0, 0, 1, 2002], # Raevskaia, French-Russian Dictionary ISBN 5-200-03040-4,
            [880, 0, 0, 0, 0, 1, 2002], # Mueler, English-Russian Dictionary ISBN 5-200-03176-1
            [331, 0, 1, 10, 0, 1, 1987], # Trademarks On Base-Metal Tableware. Eileen Woodhead,ISBN 0-660-13629-5
            [77, 1, 1, 14, 0, 2, 2012], # The Third Seeder - A Haggadah for Yom HaShoah, Irene L. Angelico & Y. Lindeman, ISBN 978-1-55065-289-5
            [134, 1, 1, 31, 0, 1, 1999], # Hebrew in 10 minutes a day, Kristine Kershil, ISBN 0-944502-25-3
            [134, 0, 0, 8, 0, 1, 1980], # American Dreams: Lost and Found, Studs Terkel, ISBN 0-394-50793-2
            [205, 1, 1, 0, 0, 2, 2010], # C# 4.0, Pocket Reference, ISBN 978-1-449-39401-1
            [93, 1, 1, 3, 0, 1, 2004], # Java Backpack Reference Guide, ISBN 978-0-321-30427-8
            [1591, 1, 1, 30, 0, 1, 2004], # Java How to Program, Deitel , ISBN 0-13-222220-5
            [505, 1, 1, 20, 0, 1, 2009], # PHP, MySQL & JavaScript, O'Reilly , ISBN 0-596-15713-4
            [654, 1, 0, 17, 0, 2, 2006], # HTML & XHTML, O'Rielly , ISBN 0-596-52732-2
            [668, 0, 0, 16, 0, 1, 2006], # Systems Development , ISBN 0-316-23256-4
            [542,1,1,18,0,1,1995], # System Analysis and Design, A project approach, David H, ISBN 0-03-097377-5
            [233,0,1,8,0,1,1993], # Implementing Client / Server Computing, A strategic Perspective, Bernard H. Boar, ISBN0-07-006215-3
            [1630, 0,1, 23,1,6, 2013], # Intermediate Accounting, ISBN 978-1-118-30085-5
            [716, 1, 0, 21,0,2,2009], # International Business, 5th Edition, ISBN 978-0-273-17654-9
            [511, 1,1,10,0,2,2018], # Canadian Tax Principles, Volume 1, ISBN 978-0-13-449820-1
            [1014,1,1,12,0,2,2018], # Canadian Tax  Principles, Volume 2, ISBN 978-0-13-479636-9
            [500,1,1,21,0,2,2018], # Canadian Tax Principles, ISBN 978-0-13-476019-3
            [266, 1, 1, 19, 1, 1, 1995], # The E-Myth, Michael E. Gerber , ISBN 978-0-88730-728-7
            [483, 1, 1, 15, 1, 1, 2013], # Lee Child, Jack Reacher - Persuader , ISBN 978-0-440-24598-8
            [483, 1, 1, 88, 2, 1, 2011], # Lee Child, Jack Reacher - The Affair , ISBN 978-0-440-24630-5
            [144, 1, 1, 18, 0, 1, 1977], # Farmer Giles of Ham, The Adventure of Tom Bombadil - ISBN 0-04-823125-8
            [250, 1, 1, 39, 0, 1, 1981], # Obasan - Joy Kogawa, ISBN 0-14-006777-9
            [112, 1, 1, 5, 0, 1, 1986], # Tom Wayman - the Face of Jack Munro, ISBN - 0-920080-59-6
            [316, 1, 0, 9, 0, 1, 1983], # Structured methods through COBOL - Robert T. Grauer, ISBN 0-13-854539-1
            [117, 1, 1, 12, 1, 1, 2014], # Star Wars - Rebels - Kogge, ISBN 978-148470270-3
            [47, 1, 1, 1, 0, 1, 2012], # Star Wars - The Clone Wars, Simon Beecroft, ISBN 978-0-7566-9247-6
            [48, 1, 1, 1, 0, 1, 2005], # Star Wars - Star Pilot, Laura Buller, ISBN 978-0-7566-1161-3
            [216, 0, 0, 55, 0, 1, 1978], # Mr. Boston's Deluxe Official Bartender's Guide
            [278, 1, 1, 18, 1, 1, 2007], # Motivational Leadership - Scott Snair, ISBN 978-1-59257-679-1
            [323, 1, 1, 3, 0, 1, 2005 ], # 2005 - The Fight for Canada - David Orchaed, ISBN 978552070253
            [25, 1, 1, 1, 1, 1, 2004] # Keys to personal Success - Don Lowry
]
labels = ['prose','prose','prose','prose','prose','prose', 'science',
          'science', 'science', 'psychology', 'psychology', 'psychology',
          'psychology', 'history', 'history', 'science',
          'psychology', 'politics', 'politics', 'politics', 'politics',
          'psychology', 'politics', 'politics', 'politics', 'politics',
          'politics', 'politics', 'politics', 'politics', 'politics',
          'politics', 'prose', 'prose', 'info. press', 'info. press',
          'prose', 'science', 'science', 'science', 'science', 'science'
          ,'text book', 'prose', 'text book', 'prose', 'prose', 'prose',
          'dictionary','prose', 'dictionary', 'dictionary', 'dictionary',
          'dictionary', 'science', 'history','language', 'politics',
          'dictionary','dictionary', 'science', 'science', 'science',
          'science', 'science', 'science' ,'science', 'science',
          'science' ,'science', 'science','psychology', 'novel', 'novel',
          'novel', 'novel', 'novel', 'science', 'novel', 'novel', 'novel',
          'science', 'science', 'novel', 'science']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[48, 1, 1, 1, 0, 1, 2005]])) # Star Wars - Star Pilot, Laura Buller, ISBN 978-0-7566-1161-3, novel
