"""Image-to-CAD corpus: a reference drawing plus the instruction to send with it.

Python rather than YAML because these cases are mostly *prose* -- an image path
and the sentence a user would actually type -- and a dict literal is easier to
add to and to diff than nested YAML.  ``harness.case_from_dict`` builds these
through the same path as the YAML cases, so the schema cannot drift.

WHY THESE EXIST ALONGSIDE ``evals/cases/*.yaml``
------------------------------------------------
The YAML cases were written for the SCRIPTED shape, where a human resolves
ambiguity on turn 2.  Their prompts therefore pre-resolve it in advance -- the
L-bracket one dictates the origin, which face the material lies on, and that a
corner block is needed.  That makes them poor tests of the unattended baseline,
which exists precisely to see what the agent does when nobody will disambiguate.

So these cases deliberately send the TERSE instruction a real user would type.
Several point at the same drawing as a YAML case on purpose: running both
measures how much of the score the pre-disambiguation was carrying.

TERSE AND GUIDED VARIANTS
-------------------------
Several drawings appear twice: once with the terse instruction a real user would
type, and once with a prompt that supplies what the IMAGE cannot carry -- the
unit, the placement convention, which of two conflicting callouts controls.  The
gap between the two scores is the measurement: it says what disambiguation is
worth, which is the number that tells you whether to invest in better prompting
or better reading.

The guided prompts state only what the reference genuinely does not determine.
They never state what it does show.  Telling the cake case "both tiers are round"
would delete the check that the agent read the plan view; telling it "the numbers
are centimetres" enables a check that was impossible without it.  That line is
the whole discipline here -- every sentence added to a guided prompt removes a
measurement, so it has to buy a better one.

GROUND TRUTH
------------
``spec``/``expect.rays`` are only present where the numbers have been read off
the drawing BY HAND and checked.  Where they are absent the case still scores --
on ``expect.dimensions`` (did it read the numbers) and ``expect.conflicts`` (did
it name a contradiction it could not resolve) -- but NOT on geometry.  Adding
ground truth is the corpus work; a case is marked ``TODO ground truth`` until
someone has done it.  Do not guess these values from a model's own output: that
is the agent grading its own homework.
"""

from __future__ import annotations

IMAGE_CASES: list[dict] = [
    {
        # Same drawing as evals/cases/lbracket_drawing.yaml, but WITHOUT the
        # placement convention spelled out.  The pair is the experiment: the
        # YAML one dictates the origin and the corner block; this one does not,
        # so the gap between their scores is what the hand-holding was worth.
        "id": "img_lbracket_terse",
        "image": "Lbracket.png",
        "prompt": "Model this bracket in BRL-CAD, named img_lbracket, "
                  "using the dimensions printed on the drawing.",
        "region": "img_lbracket",
        "expect": {
            "dimensions": ["50", "2.5", "12"],
            # The drawing does not say whether 50 mm is the inner or the outer
            # flange length, and the two readings differ by the 2.5 mm material.
            # Either is defensible; silently picking one is not.  Two groups:
            # the declaration has to name 50 AND say which reading it took, or
            # a bare "50" -- which any sane declaration mentions -- would pass
            # it for free.  Second group is the alternatives, in the forms an
            # agent actually writes them.
            "conflicts": [["50"], ["52.5", "inner", "outer"]],
            # X and Y are exactly that ambiguity (50 or 52.5), so only the
            # height is assertable -- the 50 mm on the right of the drawing runs
            # along the outer edge and means the same under either reading.
            "bbox": [None, None, 50],
            # Chosen to survive BOTH readings and any placement.  The material
            # hugs two adjacent edges of the footprint under either one, and the
            # 25 mm offsets land far from every boundary that moves.
            "rays": [
                # The defining property of an L: its footprint centre is AIR.
                # A solid block or a single plate satisfies the bbox and the
                # thickness ray below; only this separates them.
                {"desc": "footprint centre is open", "relative": True,
                 "start": [25, 25, 70], "dir": [0, 0, -1], "expect": "miss"},
                # Ø12 hole centred in the flange, so the flange's mid-height and
                # mid-length is inside it (spans 19-31 either way) and the ray
                # passes clean through the part.
                {"desc": "mounting hole is open", "relative": True,
                 "start": [25, -20, 25], "dir": [0, 1, 0], "expect": "miss"},
                # Same line, dropped below the hole: 2.5 mm of material whether
                # the flange hugs the min or the max side.
                {"desc": "flange is 2.5 mm thick", "relative": True,
                 "start": [25, -20, 10], "dir": [0, 1, 0],
                 "expect": "hit", "los": 2.5},
            ],
        },
    },
    {
        # The 2x4 brick with printed dimensions.  Genuinely self-contradictory:
        # a 6.3 mm underside cavity and a 1.0 mm top skin cannot both hold on a
        # 9.6 mm body (6.3 leaves 3.3; 1.0 needs 8.6).  A live session found this
        # unprompted and asked which controls -- the unattended run should NAME
        # it instead.  There is no right side, so which side won is not scored.
        #
        # GEOMETRY IS scored, though, on the parts the drawing DOES settle: the
        # 31.8 x 15.8 footprint, and the 2x4 stud grid (Ø4.8 on an 8 mm pitch,
        # first centre 3.9 from each edge).  Those numbers are printed on the
        # sheet, so checking them is not the agent grading itself.
        "id": "img_lego_brick_conflict",
        "image": "lego2.jpg",
        "prompt": "Draw this in BRL-CAD without any edge radii, "
                  "named img_lego_brick.",
        "region": "img_lego_brick",           # the prompt dictates the name
        "expect": {
            "dimensions": ["31.8", "15.8", "9.6", "4.8"],
            # Two GROUPS, one per side of the contradiction, because the same
            # clash has two equally correct framings: "took the 1.0 mm roof
            # over the 6.3 mm cavity" and "took an 8.6 mm cavity over the
            # 6.3 mm one" say the same thing (8.6 + 1.0 = 9.6).  Demanding the
            # literal "1.0" failed two runs that had named it perfectly well.
            "conflicts": [["6.3"], ["1.0", "8.6"]],
            # Z is deliberately NOT asserted: the body is 9.6, but the stud on
            # top is 1.8 in the side view and 1.7 in section A-A, so the overall
            # height is 11.3 or 11.4 depending on a discrepancy the drawing
            # itself has.  Asserting it would fail a defensible reading.
            "bbox": [31.8, 15.8, None],
            # Offsets from the measured minimum corner, because the prompt does
            # not say where the brick sits -- see RayCheck.relative.
            #
            # All three fire at Z = 9.6 + 0.85, i.e. INSIDE the studs and above
            # the body, where the only material is the studs themselves. That
            # height is safe for either stud reading (1.7 or 1.8), so none of
            # these rays touches the ambiguity.  Grid, from the printed numbers
            # and confirmed by arithmetic: stud centres at X = 3.9, 11.9, 19.9,
            # 27.9 (3.9 + 8k, and 31.8 - 27.9 = 3.9) and Y = 3.9, 11.9
            # (3.9 + 8 + 3.9 = 15.8).  Radius 2.4 from the Ø4.8 callout.
            "rays": [
                # Along X through the near stud row: the first thing hit is stud
                # 1, and the chord through its centre is the printed diameter.
                # A slab-topped brick would report a much longer first segment,
                # so this is the check that the studs are really 4.8 mm bosses.
                {"desc": "stud row 1 chord = diameter", "relative": True,
                 "start": [-20, 3.9, 10.45], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                {"desc": "stud row 2 chord = diameter", "relative": True,
                 "start": [-20, 11.9, 10.45], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                # Studs span X 1.5-6.3 and 9.5-14.3, so X = 7.9 is clear air the
                # whole way across.  This is what separates eight separate studs
                # from one continuous ridge -- geometry that would satisfy every
                # other check here.
                {"desc": "gap between stud columns is open", "relative": True,
                 "start": [7.9, -20, 10.45], "dir": [0, 1, 0], "expect": "miss"},
            ],
        },
    },
    {
        # The sheet shows TWO parts -- a pink 1x2 plate and a blue 1x1 brick --
        # and the prompt asks for the plate, so half the printed numbers belong
        # to the other part.  Picking the right half is the test.
        # Plate: 15.8 long, 3.2 body, studs on an 8 mm pitch with the 3.2 mm gap
        # between them (8 - 4.8) called out separately.
        "id": "img_lego_plate",
        "image": "lego.jpg",
        "prompt": "Build this plate in BRL-CAD from the drawing, "
                  "named img_lego_plate. Square edges, no fillets.",
        "region": "img_lego_plate",
        "expect": {
            "dimensions": ["15.8", "3.2", "8", "4.8"],
            # 7.8 is printed on the BRICK, not the plate, but it is the only
            # width on the sheet and the module arithmetic carries it over
            # (15.8 = 2x8 - 0.2, 7.8 = 1x8 - 0.2).  Z is left open: the plate's
            # own stud height is never dimensioned, only the brick's.
            "bbox": [15.8, 7.8, None],
            # Studs Ø4.8 at X = 3.9 and 11.9, Y = 3.9 (7.8/2), probed at
            # 3.2 + 0.85 -- inside the stud for any plausible stud height.
            "rays": [
                {"desc": "stud chord = diameter", "relative": True,
                 "start": [-20, 3.9, 4.05], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                # Studs span X 1.5-6.3 and 9.5-14.3, so 7.9 is clear air: this
                # is what distinguishes two studs from one long boss.
                {"desc": "gap between the two studs is open", "relative": True,
                 "start": [7.9, -20, 4.05], "dir": [0, 1, 0], "expect": "miss"},
            ],
        },
    },
    {
        # Rounded-edge variant: the interesting question is whether it says it
        # cannot do fillets rather than silently omitting them.  Omitting them is
        # not scored as a geometry failure -- every check here is on the studs
        # and the envelope, which fillets do not move.
        #
        # 2x4 brick, and UNLIKE lego2.jpg its stud height is unambiguous (1.8,
        # with no second value in a section view), so Z is assertable here.
        "id": "img_lego_rounded",
        "image": "lego3.jpg",
        "prompt": "Model this brick in BRL-CAD as accurately as you can, "
                  "named img_lego_rounded.",
        "region": "img_lego_rounded",
        "expect": {
            "dimensions": ["32", "16", "9.6", "4.8", "1.8"],
            "bbox": [32, 16, 11.4],           # 9.6 body + 1.8 stud
            # 8 mm pitch on a 32x16 envelope puts stud centres at X = 4, 12, 20,
            # 28 and Y = 4, 12.  Probed at 9.6 + 0.9, mid-stud.
            "rays": [
                {"desc": "stud row 1 chord = diameter", "relative": True,
                 "start": [-20, 4, 10.5], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                {"desc": "stud row 2 chord = diameter", "relative": True,
                 "start": [-20, 12, 10.5], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                # Studs span X 1.6-6.4 and 9.6-14.4; X = 8 is clear.
                {"desc": "gap between stud columns is open", "relative": True,
                 "start": [8, -20, 10.5], "dir": [0, 1, 0], "expect": "miss"},
            ],
        },
    },
    {
        # A PHOTO, not a drawing: no printed dimensions at all, so every number
        # is an assumption.  The failure to catch is inventing dimensions while
        # presenting them as read.
        #
        # Size is therefore unscoreable -- but SHAPE is not.  A 3x3x3 cube is
        # cubic whatever scale you choose, and a model that came out as a slab
        # or a single cubie is wrong on evidence the photo really does carry.
        "id": "img_rubiks_photo",
        "image": "rubic.jpg",
        "prompt": "Model this in BRL-CAD, named img_rubiks.",
        "region": "img_rubiks",
        "expect": {
            "bbox_ratio": [1, 1, 1],
            # Fractions of the measured bbox, so these hold at ANY size -- see
            # RayCheck.fraction.  Without them this case asserted only "exists"
            # and "is cubic", which a 5 mm plain block passes; it was the
            # weakest case in the corpus and contributed passes it had not
            # earned.  The grooves are the one piece of structure a photo really
            # does carry, and thirds are scale-free.
            "rays": [
                # Down the middle of a corner cubie: no groove either end, so
                # the full height of the cube.
                {"desc": "through a cubie = full height", "fraction": True,
                 "start": [1 / 6, 1 / 6, 1.5], "dir": [0, 0, -1],
                 "expect": "hit", "los_frac": 1.0},
                # Down a groove CROSSING at one third: a groove at entry and
                # another at exit, so measurably shorter. Any real 3x3 grid
                # gives < 1.0 here; a plain block gives exactly 1.0 and fails.
                {"desc": "through a groove crossing = shorter",
                 "fraction": True, "start": [1 / 3, 1 / 3, 1.5],
                 "dir": [0, 0, -1], "expect": "hit", "los_frac": 0.93},
            ],
        },
    },
    {
        # A hand sketch, and the hardest of the six -- kept because Sean asked
        # for cases that SHOULD fail and how it fails is the interesting part.
        #
        # It does carry numbers: a two-tier round cake, Ø10 x 5 on the bottom,
        # Ø6 x 5 on top (the plan view's "2" is the 2 mm ledge each side), and
        # two Ø0.5 x 2 candles.  What it does NOT carry is a UNIT, so mm and cm
        # are both defensible and an absolute bbox would fail one of them.  The
        # proportions survive either choice: 10 : 10 : (5 + 5 + 2).
        "id": "img_cake_sketch",
        "image": "birthdaycake.jpg",
        "prompt": "Make this in BRL-CAD, named img_cake.",
        "region": "img_cake",
        "expect": {
            "dimensions": ["10", "6", "5", "2", "0.5"],
            "bbox_ratio": [10, 10, 12],
            # Scale-free, like the Rubik's rays: fractions of the measured bbox.
            # A ratio alone could not tell a round cake from a square one, nor
            # a two-tier cake from a plain cylinder -- both of which have the
            # same 10:10:12 envelope. These say it is round AND stepped without
            # ever needing the unit the sketch withholds.
            "rays": [
                # 5% in from the footprint corner is 63.6% of the half-width
                # from the axis -- outside a tier that fills the footprint, so
                # air for a round cake and material for a square one.
                {"desc": "footprint corner is empty (tiers are round)",
                 "fraction": True, "start": [0.05, 0.05, 1.5],
                 "dir": [0, 0, -1], "expect": "miss"},
                # Over the base but OUTSIDE the narrower top tier (which spans
                # 0.2-0.8 of the footprint): the ray should stop at the base's
                # top, i.e. 5 of the 12 units of total height.
                {"desc": "step: base tier only, not full height",
                 "fraction": True, "start": [0.1, 0.5, 1.5], "dir": [0, 0, -1],
                 "expect": "hit", "los_frac": 5 / 12},
            ],
        },
    },

    # --- guided twins -----------------------------------------------------
    #
    # Same drawings, prompts that supply only what the image cannot carry.
    # Each one buys absolute geometry where the terse twin could assert only
    # proportions or a single axis.
    {
        # Terse twin: img_lego_brick_conflict, which can assert X and Y only,
        # because 1.7-vs-1.8 leaves the height open and the 6.3-vs-1.0 clash
        # leaves the interior open.  Naming section A-A as controlling settles
        # both -- and makes the hollow-stud failure unambiguous rather than a
        # defensible alternative reading.
        "id": "img_lego_brick_guided",
        "image": "lego2.jpg",
        "prompt": "Draw this in BRL-CAD without any edge radii, named "
                  "img_lego_brick_g. Section A-A controls the interior: the top "
                  "skin is 1.0 mm and the studs are solid all the way through. "
                  "Where the stud height differs between views, take the 1.8 mm "
                  "from the side view.",
        "region": "img_lego_brick_g",
        "expect": {
            "dimensions": ["31.8", "15.8", "9.6", "4.8"],
            "bbox": [31.8, 15.8, 11.4],       # Z assertable now: 9.6 + 1.8
            "rays": [
                {"desc": "stud row 1 chord = diameter", "relative": True,
                 "start": [-20, 3.9, 10.45], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                {"desc": "stud row 2 chord = diameter", "relative": True,
                 "start": [-20, 11.9, 10.45], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                {"desc": "gap between stud columns is open", "relative": True,
                 "start": [7.9, -20, 10.45], "dir": [0, 1, 0], "expect": "miss"},
            ],
        },
    },
    {
        # Terse twin: img_lego_plate.  The width and the plate's own stud height
        # are the two numbers the sheet never prints for THIS part, so supplying
        # them is what turns the envelope into an absolute check.
        "id": "img_lego_plate_guided",
        "image": "lego.jpg",
        "prompt": "Build the pink plate from this drawing in BRL-CAD, named "
                  "img_lego_plate_g. Square edges, no fillets. The plate is one "
                  "stud wide -- 7.8 mm, the same width as the brick beside it -- "
                  "and its studs are 1.7 mm tall like the brick's.",
        "region": "img_lego_plate_g",
        "expect": {
            "dimensions": ["15.8", "7.8", "3.2", "8", "4.8"],
            "bbox": [15.8, 7.8, 4.9],         # 3.2 body + 1.7 stud
            "rays": [
                {"desc": "stud chord = diameter", "relative": True,
                 "start": [-20, 3.9, 4.05], "dir": [1, 0, 0],
                 "expect": "hit", "los": 4.8},
                {"desc": "gap between the two studs is open", "relative": True,
                 "start": [7.9, -20, 4.05], "dir": [0, 1, 0], "expect": "miss"},
            ],
        },
    },
    {
        # Terse twin: img_rubiks_photo, which can only say "cubic".  A photo
        # cannot carry a scale at all, so the size has to be given outright; the
        # groove geometry is given because the photo shows grooves but no depth.
        "id": "img_rubiks_guided",
        "image": "rubic.jpg",
        "prompt": "Model this puzzle cube in BRL-CAD, named img_rubiks_g. It is "
                  "57 mm on every side. Cut the groove lines in as square "
                  "channels 2 mm wide and 2 mm deep, running the full width of "
                  "each face and dividing every face into an even 3x3 grid. "
                  "Sharp edges, no rounding.",
        "region": "img_rubiks_g",
        "expect": {
            "dimensions": ["57", "2"],
            "bbox": [57, 57, 57],
            # Grooves fall at 19 and 38 mm.  Down the middle of a cubie the
            # cube is its full 57 mm thick; down a groove CROSSING it loses
            # 2 mm at entry and 2 mm at exit.  The pair is what proves the
            # channels were actually cut rather than painted on in colour.
            "rays": [
                {"desc": "through a cubie face = full thickness",
                 "relative": True, "start": [9.5, 9.5, 80], "dir": [0, 0, -1],
                 "expect": "hit", "los": 57},
                {"desc": "through a groove crossing = 2 mm shallower each end",
                 "relative": True, "start": [19, 19, 80], "dir": [0, 0, -1],
                 "expect": "hit", "los": 53},
            ],
        },
    },
    {
        # Terse twin: img_cake_sketch, which can only assert proportions because
        # the sketch has no unit.  Supplying the unit and the placement is
        # enough to make every dimension absolute -- and, crucially, to make
        # ROUNDNESS testable, which no ratio can express.  The prompt does not
        # say the tiers are round: reading that off the plan view is the test.
        "id": "img_cake_guided",
        "image": "birthdaycake.jpg",
        "prompt": "Make this in BRL-CAD, named img_cake_g. The numbers on the "
                  "sketch are centimetres. Put the base centred on the origin "
                  "with the axis along +Z.",
        "region": "img_cake_g",
        "expect": {
            "dimensions": ["10", "6", "5", "2"],
            "bbox": [100, 100, 120],          # cm -> mm, 5 + 5 + 2 tall
            "rays": [
                # A Ø100 tier centred in a 100x100 footprint leaves the corners
                # empty: 5 mm in from the corner is 63.6 mm from the axis, well
                # outside. A square cake hits here. This is the check the terse
                # twin cannot make at all, because it needs a length.
                {"desc": "footprint corner is empty (tiers are round)",
                 "relative": True, "start": [5, 5, 150], "dir": [0, 0, -1],
                 "expect": "miss"},
                # Across the upper tier, which the sketch puts at 6 wide.
                {"desc": "top tier is 60 mm across", "relative": True,
                 "start": [-40, 50, 70], "dir": [1, 0, 0],
                 "expect": "hit", "los": 60},
            ],
        },
    },
]
