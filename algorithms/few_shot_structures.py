



GSM8K_FEW_SHOTS = [
    """{
        "steps": [
            "Natalia sold 48 clips in April.",
            "She sold half as many clips in May, which is 24.",
            "Adding April and May sales gives 48 + 24."
        ],
        "answer": "72"
    }""",

    """{
        "steps": [
            "Babysitting rate is 12 dollars per hour.",
            "50 minutes equals 5/6 of an hour.",
            "Multiply 12 by 5/6 to get earnings."
        ],
        "answer": "10"
    }""",

    """{
        "steps": [
            "The wallet costs 100 dollars.",
            "Betty has half the money, which is 50.",
            "Her parents give 15 dollars and her grandparents give 30 dollars.",
            "Total money becomes 95, so subtract from 100."
        ],
        "answer": "5"
    }""",

    """{
        "steps": [
            "The book has 120 pages total.",
            "Julie read 12 pages yesterday and 24 pages today.",
            "She has read 36 pages, leaving 84 pages.",
            "Half of the remaining pages is 42."
        ],
        "answer": "42"
    }"""
]


MATH500_FEW_SHOTS = [
    """{
        "steps": [
            "Continuity at $x=2$ requires $a(2)+3 = 2-5$.",
            "Solving gives $2a+3=-3$ so $a=-3$.",
            "Continuity at $x=-2$ requires $2(-2)-b = -2-5$.",
            "Solving gives $-4-b=-7$ so $b=3$.",
            "Thus $a+b=-3+3$."
        ],
        "answer": "0"
    }""",

    """{
        "steps": [
            "Let the original formation have $m$ members per row and $r$ rows, so the total is $mr+2$.",
            "The new formation has $(m+1)$ members per row and $(r-2)$ rows, so $(m+1)(r-2)=mr+2$.",
            "Expanding gives $mr-2m+r-2=mr+2$ which simplifies to $r=2m+4$.",
            "Total members are $N=mr+2=m(2m+4)+2=2m^2+4m+2$.",
            "Impose $N<100$ and choose the largest valid integer.",
            "This yields $N=98$."
        ],
        "answer": "98"
    }""",

    """{
        "steps": [
            "Combine constant terms $4+100+9$ into a single constant.",
            "Identify the highest power of $x$ in the polynomial.",
            "The highest-degree terms are $2\\pi x^4$ and $\\sqrt{10}x^4$.",
            "Therefore the degree of the polynomial is $4$."
        ],
        "answer": "4"
    }""",

    """{
        "steps": [
            "Let $w$ be the number of days Sam worked.",
            "He did not work $20-w$ days.",
            "Total earnings are $60w-30(20-w)=660$.",
            "Solving gives $90w=1260$ so $w=14$.",
            "Days not worked are $20-14$."
        ],
        "answer": "6"
    }"""
]



GSM_SYMBOLIC_FEW_SHOTS = [
    "<<tf - t>>",
    "<<c + nc>>",
    "<<ch1 + ch2 - a>>",
    "<<l1 - g>>",
    "<<t + tm + td>>",
    "<<c + nc * (d2 - d1 + 1)>>",
    "<<gb1 - l1 - l2>>",
    "<<m - q * p>>"
]



PROVER9_FEW_SHOTS = [
    """Predicates:
Dependent(x) ::: x is a person dependent on caffeine.
Drinks(x) ::: x regularly drinks coffee.
Jokes(x) ::: x jokes about being addicted to caffeine.
Unaware(x) ::: x is unaware that caffeine is a drug.
Student(x) ::: x is a student.

Premises:
{forall} x (Drinks(x) {implies} Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
{forall} x (Drinks(x) {xor} Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
{forall} x (Jokes(x) {implies} {not}Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.
(Student(rina) {and} Unaware(rina)) {xor} {not}(Student(rina) {or} Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.

Conclusion:
Jokes(rina) {xor} Unaware(rina) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
------""",

    """Predicates:
Czech(x) ::: x is a Czech person.
ChoralConductor(x) ::: x is a choral conductor.
Musician(x) ::: x is a musician.
Love(x, y) ::: x loves y.
Author(x, y) ::: x is the author of y.
Book(x) ::: x is a book.
Publish(x, y) ::: x is published in year y.
Specialize(x, y) ::: x specializes in y.

Premises:
Czech(miroslav) {and} ChoralConductor(miroslav) {and} Specialize(miroslav, renaissance) {and} Specialize(miroslav, baroque) ::: Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music.
{forall} x (ChoralConductor(x) {implies} Musician(x)) ::: Any choral conductor is a musician.
{exists} x (Musician(x) {and} Love(x, music)) ::: Some musicians love music.
Book(methodOfStudyingGregorianChant) {and} Author(miroslav, methodOfStudyingGregorianChant) {and} Publish(methodOfStudyingGregorianChant, year1946) ::: Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.

Conclusion:
Love(miroslav, music) ::: Miroslav Venhoda loved music.
------""",

    """Predicates:
Mammal(x) ::: x is a mammal.
Whale(x) ::: x is a whale.
Fish(x) ::: x is a fish.
LivesInWater(x) ::: x lives in water.

Premises:
{forall} x (Whale(x) {implies} Mammal(x)) ::: All whales are mammals.
{forall} x (Whale(x) {implies} LivesInWater(x)) ::: All whales live in water.
Whale(willy) ::: Willy is a whale.
{forall} x (Fish(x) {implies} LivesInWater(x)) ::: All fish live in water.
{forall} x (Mammal(x) {implies} {not}Fish(x)) ::: No mammal is a fish.

Conclusion:
{not}Fish(willy) ::: Willy is not a fish.
------""",

    """Predicates:
Even(x) ::: x is even.
Odd(x) ::: x is odd.
Integer(x) ::: x is an integer.
Sum(x, y, z) ::: z is the sum of x and y.

Premises:
Integer(a) {and} Integer(b) ::: a and b are integers.
Even(a) ::: a is even.
Odd(b) ::: b is odd.
{forall} x ({forall} y ((Even(x) {and} Odd(y)) {implies} Odd(x+y))) ::: The sum of an even integer and an odd integer is odd.

Conclusion:
Odd(a+b) ::: a+b is odd.
------"""
]
