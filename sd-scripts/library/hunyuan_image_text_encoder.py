import json
import re
from typing import Tuple, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    T5Tokenizer,
)
from transformers.models.t5.modeling_t5 import T5Stack
from accelerate import init_empty_weights

from library.safetensors_utils import load_safetensors
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


BYT5_TOKENIZER_PATH = "google/byt5-small"
QWEN_2_5_VL_IMAGE_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


# Copy from Glyph-SDXL-V2

COLOR_IDX_JSON = """{"white": 0, "black": 1, "darkslategray": 2, "dimgray": 3, "darkolivegreen": 4, "midnightblue": 5, "saddlebrown": 6, "sienna": 7, "whitesmoke": 8, "darkslateblue": 9, 
"indianred": 10, "linen": 11, "maroon": 12, "khaki": 13, "sandybrown": 14, "gray": 15, "gainsboro": 16, "teal": 17, "peru": 18, "gold": 19, 
"snow": 20, "firebrick": 21, "crimson": 22, "chocolate": 23, "tomato": 24, "brown": 25, "goldenrod": 26, "antiquewhite": 27, "rosybrown": 28, "steelblue": 29, 
"floralwhite": 30, "seashell": 31, "darkgreen": 32, "oldlace": 33, "darkkhaki": 34, "burlywood": 35, "red": 36, "darkgray": 37, "orange": 38, "royalblue": 39, 
"seagreen": 40, "lightgray": 41, "tan": 42, "coral": 43, "beige": 44, "palevioletred": 45, "wheat": 46, "lavender": 47, "darkcyan": 48, "slateblue": 49, 
"slategray": 50, "orangered": 51, "silver": 52, "olivedrab": 53, "forestgreen": 54, "darkgoldenrod": 55, "ivory": 56, "darkorange": 57, "yellow": 58, "hotpink": 59, 
"ghostwhite": 60, "lightcoral": 61, "indigo": 62, "bisque": 63, "darkred": 64, "darksalmon": 65, "lightslategray": 66, "dodgerblue": 67, "lightpink": 68, "mistyrose": 69, 
"mediumvioletred": 70, "cadetblue": 71, "deeppink": 72, "salmon": 73, "palegoldenrod": 74, "blanchedalmond": 75, "lightseagreen": 76, "cornflowerblue": 77, "yellowgreen": 78, "greenyellow": 79, 
"navajowhite": 80, "papayawhip": 81, "mediumslateblue": 82, "purple": 83, "blueviolet": 84, "pink": 85, "cornsilk": 86, "lightsalmon": 87, "mediumpurple": 88, "moccasin": 89, 
"turquoise": 90, "mediumseagreen": 91, "lavenderblush": 92, "mediumblue": 93, "darkseagreen": 94, "mediumturquoise": 95, "paleturquoise": 96, "skyblue": 97, "lemonchiffon": 98, "olive": 99, 
"peachpuff": 100, "lightyellow": 101, "lightsteelblue": 102, "mediumorchid": 103, "plum": 104, "darkturquoise": 105, "aliceblue": 106, "mediumaquamarine": 107, "orchid": 108, "powderblue": 109, 
"blue": 110, "darkorchid": 111, "violet": 112, "lightskyblue": 113, "lightcyan": 114, "lightgoldenrodyellow": 115, "navy": 116, "thistle": 117, "honeydew": 118, "mintcream": 119, 
"lightblue": 120, "darkblue": 121, "darkmagenta": 122, "deepskyblue": 123, "magenta": 124, "limegreen": 125, "darkviolet": 126, "cyan": 127, "palegreen": 128, "aquamarine": 129, 
"lawngreen": 130, "lightgreen": 131, "azure": 132, "chartreuse": 133, "green": 134, "mediumspringgreen": 135, "lime": 136, "springgreen": 137}"""

MULTILINGUAL_10_LANG_IDX_JSON = """{"en-Montserrat-Regular": 0, "en-Poppins-Italic": 1, "en-GlacialIndifference-Regular": 2, "en-OpenSans-ExtraBoldItalic": 3, "en-Montserrat-Bold": 4, "en-Now-Regular": 5, "en-Garet-Regular": 6, "en-LeagueSpartan-Bold": 7, "en-DMSans-Regular": 8, "en-OpenSauceOne-Regular": 9, 
"en-OpenSans-ExtraBold": 10, "en-KGPrimaryPenmanship": 11, "en-Anton-Regular": 12, "en-Aileron-BlackItalic": 13, "en-Quicksand-Light": 14, "en-Roboto-BoldItalic": 15, "en-TheSeasons-It": 16, "en-Kollektif": 17, "en-Inter-BoldItalic": 18, "en-Poppins-Medium": 19, 
"en-Poppins-Light": 20, "en-RoxboroughCF-RegularItalic": 21, "en-PlayfairDisplay-SemiBold": 22, "en-Agrandir-Italic": 23, "en-Lato-Regular": 24, "en-MoreSugarRegular": 25, "en-CanvaSans-RegularItalic": 26, "en-PublicSans-Italic": 27, "en-CodePro-NormalLC": 28, "en-Belleza-Regular": 29, 
"en-JosefinSans-Bold": 30, "en-HKGrotesk-Bold": 31, "en-Telegraf-Medium": 32, "en-BrittanySignatureRegular": 33, "en-Raleway-ExtraBoldItalic": 34, "en-Mont-RegularItalic": 35, "en-Arimo-BoldItalic": 36, "en-Lora-Italic": 37, "en-ArchivoBlack-Regular": 38, "en-Poppins": 39, 
"en-Barlow-Black": 40, "en-CormorantGaramond-Bold": 41, "en-LibreBaskerville-Regular": 42, "en-CanvaSchoolFontRegular": 43, "en-BebasNeueBold": 44, "en-LazydogRegular": 45, "en-FredokaOne-Regular": 46, "en-Horizon-Bold": 47, "en-Nourd-Regular": 48, "en-Hatton-Regular": 49, 
"en-Nunito-ExtraBoldItalic": 50, "en-CerebriSans-Regular": 51, "en-Montserrat-Light": 52, "en-TenorSans": 53, "en-Norwester-Regular": 54, "en-ClearSans-Bold": 55, "en-Cardo-Regular": 56, "en-Alice-Regular": 57, "en-Oswald-Regular": 58, "en-Gaegu-Bold": 59, 
"en-Muli-Black": 60, "en-TAN-PEARL-Regular": 61, "en-CooperHewitt-Book": 62, "en-Agrandir-Grand": 63, "en-BlackMango-Thin": 64, "en-DMSerifDisplay-Regular": 65, "en-Antonio-Bold": 66, "en-Sniglet-Regular": 67, "en-BeVietnam-Regular": 68, "en-NunitoSans10pt-BlackItalic": 69, 
"en-AbhayaLibre-ExtraBold": 70, "en-Rubik-Regular": 71, "en-PPNeueMachina-Regular": 72, "en-TAN - MON CHERI-Regular": 73, "en-Jua-Regular": 74, "en-Playlist-Script": 75, "en-SourceSansPro-BoldItalic": 76, "en-MoonTime-Regular": 77, "en-Eczar-ExtraBold": 78, "en-Gatwick-Regular": 79, 
"en-MonumentExtended-Regular": 80, "en-BarlowSemiCondensed-Regular": 81, "en-BarlowCondensed-Regular": 82, "en-Alegreya-Regular": 83, "en-DreamAvenue": 84, "en-RobotoCondensed-Italic": 85, "en-BobbyJones-Regular": 86, "en-Garet-ExtraBold": 87, "en-YesevaOne-Regular": 88, "en-Dosis-ExtraBold": 89, 
"en-LeagueGothic-Regular": 90, "en-OpenSans-Italic": 91, "en-TANAEGEAN-Regular": 92, "en-Maharlika-Regular": 93, "en-MarykateRegular": 94, "en-Cinzel-Regular": 95, "en-Agrandir-Wide": 96, "en-Chewy-Regular": 97, "en-BodoniFLF-BoldItalic": 98, "en-Nunito-BlackItalic": 99, 
"en-LilitaOne": 100, "en-HandyCasualCondensed-Regular": 101, "en-Ovo": 102, "en-Livvic-Regular": 103, "en-Agrandir-Narrow": 104, "en-CrimsonPro-Italic": 105, "en-AnonymousPro-Bold": 106, "en-NF-OneLittleFont-Bold": 107, "en-RedHatDisplay-BoldItalic": 108, "en-CodecPro-Regular": 109, 
"en-HalimunRegular": 110, "en-LibreFranklin-Black": 111, "en-TeXGyreTermes-BoldItalic": 112, "en-Shrikhand-Regular": 113, "en-TTNormsPro-Italic": 114, "en-Gagalin-Regular": 115, "en-OpenSans-Bold": 116, "en-GreatVibes-Regular": 117, "en-Breathing": 118, "en-HeroLight-Regular": 119, 
"en-KGPrimaryDots": 120, "en-Quicksand-Bold": 121, "en-Brice-ExtraLightSemiExpanded": 122, "en-Lato-BoldItalic": 123, "en-Fraunces9pt-Italic": 124, "en-AbrilFatface-Regular": 125, "en-BerkshireSwash-Regular": 126, "en-Atma-Bold": 127, "en-HolidayRegular": 128, "en-BebasNeueCyrillic": 129, 
"en-IntroRust-Base": 130, "en-Gistesy": 131, "en-BDScript-Regular": 132, "en-ApricotsRegular": 133, "en-Prompt-Black": 134, "en-TAN MERINGUE": 135, "en-Sukar Regular": 136, "en-GentySans-Regular": 137, "en-NeueEinstellung-Normal": 138, "en-Garet-Bold": 139, 
"en-FiraSans-Black": 140, "en-BantayogLight": 141, "en-NotoSerifDisplay-Black": 142, "en-TTChocolates-Regular": 143, "en-Ubuntu-Regular": 144, "en-Assistant-Bold": 145, "en-ABeeZee-Regular": 146, "en-LexendDeca-Regular": 147, "en-KingredSerif": 148, "en-Radley-Regular": 149, 
"en-BrownSugar": 150, "en-MigraItalic-ExtraboldItalic": 151, "en-ChildosArabic-Regular": 152, "en-PeaceSans": 153, "en-LondrinaSolid-Black": 154, "en-SpaceMono-BoldItalic": 155, "en-RobotoMono-Light": 156, "en-CourierPrime-Regular": 157, "en-Alata-Regular": 158, "en-Amsterdam-One": 159, 
"en-IreneFlorentina-Regular": 160, "en-CatchyMager": 161, "en-Alta_regular": 162, "en-ArticulatCF-Regular": 163, "en-Raleway-Regular": 164, "en-BrasikaDisplay": 165, "en-TANAngleton-Italic": 166, "en-NotoSerifDisplay-ExtraCondensedItalic": 167, "en-Bryndan Write": 168, "en-TTCommonsPro-It": 169, 
"en-AlexBrush-Regular": 170, "en-Antic-Regular": 171, "en-TTHoves-Bold": 172, "en-DroidSerif": 173, "en-AblationRegular": 174, "en-Marcellus-Regular": 175, "en-Sanchez-Italic": 176, "en-JosefinSans": 177, "en-Afrah-Regular": 178, "en-PinyonScript": 179, 
"en-TTInterphases-BoldItalic": 180, "en-Yellowtail-Regular": 181, "en-Gliker-Regular": 182, "en-BobbyJonesSoft-Regular": 183, "en-IBMPlexSans": 184, "en-Amsterdam-Three": 185, "en-Amsterdam-FourSlant": 186, "en-TTFors-Regular": 187, "en-Quattrocento": 188, "en-Sifonn-Basic": 189, 
"en-AlegreyaSans-Black": 190, "en-Daydream": 191, "en-AristotelicaProTx-Rg": 192, "en-NotoSerif": 193, "en-EBGaramond-Italic": 194, "en-HammersmithOne-Regular": 195, "en-RobotoSlab-Regular": 196, "en-DO-Sans-Regular": 197, "en-KGPrimaryDotsLined": 198, "en-Blinker-Regular": 199, 
"en-TAN NIMBUS": 200, "en-Blueberry-Regular": 201, "en-Rosario-Regular": 202, "en-Forum": 203, "en-MistrullyRegular": 204, "en-SourceSerifPro-Regular": 205, "en-Bugaki-Regular": 206, "en-CMUSerif-Roman": 207, "en-GulfsDisplay-NormalItalic": 208, "en-PTSans-Bold": 209, 
"en-Sensei-Medium": 210, "en-SquadaOne-Regular": 211, "en-Arapey-Italic": 212, "en-Parisienne-Regular": 213, "en-Aleo-Italic": 214, "en-QuicheDisplay-Italic": 215, "en-RocaOne-It": 216, "en-Funtastic-Regular": 217, "en-PTSerif-BoldItalic": 218, "en-Muller-RegularItalic": 219, 
"en-ArgentCF-Regular": 220, "en-Brightwall-Italic": 221, "en-Knewave-Regular": 222, "en-TYSerif-D": 223, "en-Agrandir-Tight": 224, "en-AlfaSlabOne-Regular": 225, "en-TANTangkiwood-Display": 226, "en-Kief-Montaser-Regular": 227, "en-Gotham-Book": 228, "en-JuliusSansOne-Regular": 229, 
"en-CocoGothic-Italic": 230, "en-SairaCondensed-Regular": 231, "en-DellaRespira-Regular": 232, "en-Questrial-Regular": 233, "en-BukhariScript-Regular": 234, "en-HelveticaWorld-Bold": 235, "en-TANKINDRED-Display": 236, "en-CinzelDecorative-Regular": 237, "en-Vidaloka-Regular": 238, "en-AlegreyaSansSC-Black": 239, 
"en-FeelingPassionate-Regular": 240, "en-QuincyCF-Regular": 241, "en-FiraCode-Regular": 242, "en-Genty-Regular": 243, "en-Nickainley-Normal": 244, "en-RubikOne-Regular": 245, "en-Gidole-Regular": 246, "en-Borsok": 247, "en-Gordita-RegularItalic": 248, "en-Scripter-Regular": 249, 
"en-Buffalo-Regular": 250, "en-KleinText-Regular": 251, "en-Creepster-Regular": 252, "en-Arvo-Bold": 253, "en-GabrielSans-NormalItalic": 254, "en-Heebo-Black": 255, "en-LexendExa-Regular": 256, "en-BrixtonSansTC-Regular": 257, "en-GildaDisplay-Regular": 258, "en-ChunkFive-Roman": 259, 
"en-Amaranth-BoldItalic": 260, "en-BubbleboddyNeue-Regular": 261, "en-MavenPro-Bold": 262, "en-TTDrugs-Italic": 263, "en-CyGrotesk-KeyRegular": 264, "en-VarelaRound-Regular": 265, "en-Ruda-Black": 266, "en-SafiraMarch": 267, "en-BloggerSans": 268, "en-TANHEADLINE-Regular": 269, 
"en-SloopScriptPro-Regular": 270, "en-NeueMontreal-Regular": 271, "en-Schoolbell-Regular": 272, "en-SigherRegular": 273, "en-InriaSerif-Regular": 274, "en-JetBrainsMono-Regular": 275, "en-MADEEvolveSans": 276, "en-Dekko": 277, "en-Handyman-Regular": 278, "en-Aileron-BoldItalic": 279, 
"en-Bright-Italic": 280, "en-Solway-Regular": 281, "en-Higuen-Regular": 282, "en-WedgesItalic": 283, "en-TANASHFORD-BOLD": 284, "en-IBMPlexMono": 285, "en-RacingSansOne-Regular": 286, "en-RegularBrush": 287, "en-OpenSans-LightItalic": 288, "en-SpecialElite-Regular": 289, 
"en-FuturaLTPro-Medium": 290, "en-MaragsaDisplay": 291, "en-BigShouldersDisplay-Regular": 292, "en-BDSans-Regular": 293, "en-RasputinRegular": 294, "en-Yvesyvesdrawing-BoldItalic": 295, "en-Bitter-Regular": 296, "en-LuckiestGuy-Regular": 297, "en-CanvaSchoolFontDotted": 298, "en-TTFirsNeue-Italic": 299, 
"en-Sunday-Regular": 300, "en-HKGothic-MediumItalic": 301, "en-CaveatBrush-Regular": 302, "en-HeliosExt": 303, "en-ArchitectsDaughter-Regular": 304, "en-Angelina": 305, "en-Calistoga-Regular": 306, "en-ArchivoNarrow-Regular": 307, "en-ObjectSans-MediumSlanted": 308, "en-AyrLucidityCondensed-Regular": 309, 
"en-Nexa-RegularItalic": 310, "en-Lustria-Regular": 311, "en-Amsterdam-TwoSlant": 312, "en-Virtual-Regular": 313, "en-Brusher-Regular": 314, "en-NF-Lepetitcochon-Regular": 315, "en-TANTWINKLE": 316, "en-LeJour-Serif": 317, "en-Prata-Regular": 318, "en-PPWoodland-Regular": 319, 
"en-PlayfairDisplay-BoldItalic": 320, "en-AmaticSC-Regular": 321, "en-Cabin-Regular": 322, "en-Manjari-Bold": 323, "en-MrDafoe-Regular": 324, "en-TTRamillas-Italic": 325, "en-Luckybones-Bold": 326, "en-DarkerGrotesque-Light": 327, "en-BellabooRegular": 328, "en-CormorantSC-Bold": 329, 
"en-GochiHand-Regular": 330, "en-Atteron": 331, "en-RocaTwo-Lt": 332, "en-ZCOOLXiaoWei-Regular": 333, "en-TANSONGBIRD": 334, "en-HeadingNow-74Regular": 335, "en-Luthier-BoldItalic": 336, "en-Oregano-Regular": 337, "en-AyrTropikaIsland-Int": 338, "en-Mali-Regular": 339, 
"en-DidactGothic-Regular": 340, "en-Lovelace-Regular": 341, "en-BakerieSmooth-Regular": 342, "en-CarterOne": 343, "en-HussarBd": 344, "en-OldStandard-Italic": 345, "en-TAN-ASTORIA-Display": 346, "en-rugratssans-Regular": 347, "en-BMHANNA": 348, "en-BetterSaturday": 349, 
"en-AdigianaToybox": 350, "en-Sailors": 351, "en-PlayfairDisplaySC-Italic": 352, "en-Etna-Regular": 353, "en-Revive80Signature": 354, "en-CAGenerated": 355, "en-Poppins-Regular": 356, "en-Jonathan-Regular": 357, "en-Pacifico-Regular": 358, "en-Saira-Black": 359, 
"en-Loubag-Regular": 360, "en-Decalotype-Black": 361, "en-Mansalva-Regular": 362, "en-Allura-Regular": 363, "en-ProximaNova-Bold": 364, "en-TANMIGNON-DISPLAY": 365, "en-ArsenicaAntiqua-Regular": 366, "en-BreulGroteskA-RegularItalic": 367, "en-HKModular-Bold": 368, "en-TANNightingale-Regular": 369, 
"en-AristotelicaProCndTxt-Rg": 370, "en-Aprila-Regular": 371, "en-Tomorrow-Regular": 372, "en-AngellaWhite": 373, "en-KaushanScript-Regular": 374, "en-NotoSans": 375, "en-LeJour-Script": 376, "en-BrixtonTC-Regular": 377, "en-OleoScript-Regular": 378, "en-Cakerolli-Regular": 379, 
"en-Lobster-Regular": 380, "en-FrunchySerif-Regular": 381, "en-PorcelainRegular": 382, "en-AlojaExtended": 383, "en-SergioTrendy-Italic": 384, "en-LovelaceText-Bold": 385, "en-Anaktoria": 386, "en-JimmyScript-Light": 387, "en-IBMPlexSerif": 388, "en-Marta": 389, 
"en-Mango-Regular": 390, "en-Overpass-Italic": 391, "en-Hagrid-Regular": 392, "en-ElikaGorica": 393, "en-Amiko-Regular": 394, "en-EFCOBrookshire-Regular": 395, "en-Caladea-Regular": 396, "en-MoonlightBold": 397, "en-Staatliches-Regular": 398, "en-Helios-Bold": 399, 
"en-Satisfy-Regular": 400, "en-NexaScript-Regular": 401, "en-Trocchi-Regular": 402, "en-March": 403, "en-IbarraRealNova-Regular": 404, "en-Nectarine-Regular": 405, "en-Overpass-Light": 406, "en-TruetypewriterPolyglOTT": 407, "en-Bangers-Regular": 408, "en-Lazord-BoldExpandedItalic": 409, 
"en-Chloe-Regular": 410, "en-BaskervilleDisplayPT-Regular": 411, "en-Bright-Regular": 412, "en-Vollkorn-Regular": 413, "en-Harmattan": 414, "en-SortsMillGoudy-Regular": 415, "en-Biryani-Bold": 416, "en-SugoProDisplay-Italic": 417, "en-Lazord-BoldItalic": 418, "en-Alike-Regular": 419, 
"en-PermanentMarker-Regular": 420, "en-Sacramento-Regular": 421, "en-HKGroteskPro-Italic": 422, "en-Aleo-BoldItalic": 423, "en-Noot": 424, "en-TANGARLAND-Regular": 425, "en-Twister": 426, "en-Arsenal-Italic": 427, "en-Bogart-Italic": 428, "en-BethEllen-Regular": 429, 
"en-Caveat-Regular": 430, "en-BalsamiqSans-Bold": 431, "en-BreeSerif-Regular": 432, "en-CodecPro-ExtraBold": 433, "en-Pierson-Light": 434, "en-CyGrotesk-WideRegular": 435, "en-Lumios-Marker": 436, "en-Comfortaa-Bold": 437, "en-TraceFontRegular": 438, "en-RTL-AdamScript-Regular": 439, 
"en-EastmanGrotesque-Italic": 440, "en-Kalam-Bold": 441, "en-ChauPhilomeneOne-Regular": 442, "en-Coiny-Regular": 443, "en-Lovera": 444, "en-Gellatio": 445, "en-TitilliumWeb-Bold": 446, "en-OilvareBase-Italic": 447, "en-Catamaran-Black": 448, "en-Anteb-Italic": 449, 
"en-SueEllenFrancisco": 450, "en-SweetApricot": 451, "en-BrightSunshine": 452, "en-IM_FELL_Double_Pica_Italic": 453, "en-Granaina-limpia": 454, "en-TANPARFAIT": 455, "en-AcherusGrotesque-Regular": 456, "en-AwesomeLathusca-Italic": 457, "en-Signika-Bold": 458, "en-Andasia": 459, 
"en-DO-AllCaps-Slanted": 460, "en-Zenaida-Regular": 461, "en-Fahkwang-Regular": 462, "en-Play-Regular": 463, "en-BERNIERRegular-Regular": 464, "en-PlumaThin-Regular": 465, "en-SportsWorld": 466, "en-Garet-Black": 467, "en-CarolloPlayscript-BlackItalic": 468, "en-Cheque-Regular": 469, 
"en-SEGO": 470, "en-BobbyJones-Condensed": 471, "en-NexaSlab-RegularItalic": 472, "en-DancingScript-Regular": 473, "en-PaalalabasDisplayWideBETA": 474, "en-Magnolia-Script": 475, "en-OpunMai-400It": 476, "en-MadelynFill-Regular": 477, "en-ZingRust-Base": 478, "en-FingerPaint-Regular": 479, 
"en-BostonAngel-Light": 480, "en-Gliker-RegularExpanded": 481, "en-Ahsing": 482, "en-Engagement-Regular": 483, "en-EyesomeScript": 484, "en-LibraSerifModern-Regular": 485, "en-London-Regular": 486, "en-AtkinsonHyperlegible-Regular": 487, "en-StadioNow-TextItalic": 488, "en-Aniyah": 489, 
"en-ITCAvantGardePro-Bold": 490, "en-Comica-Regular": 491, "en-Coustard-Regular": 492, "en-Brice-BoldCondensed": 493, "en-TANNEWYORK-Bold": 494, "en-TANBUSTER-Bold": 495, "en-Alatsi-Regular": 496, "en-TYSerif-Book": 497, "en-Jingleberry": 498, "en-Rajdhani-Bold": 499, 
"en-LobsterTwo-BoldItalic": 500, "en-BestLight-Medium": 501, "en-Hitchcut-Regular": 502, "en-GermaniaOne-Regular": 503, "en-Emitha-Script": 504, "en-LemonTuesday": 505, "en-Cubao_Free_Regular": 506, "en-MonterchiSerif-Regular": 507, "en-AllertaStencil-Regular": 508, "en-RTL-Sondos-Regular": 509, 
"en-HomemadeApple-Regular": 510, "en-CosmicOcto-Medium": 511, "cn-HelloFont-FangHuaTi": 0, "cn-HelloFont-ID-DianFangSong-Bold": 1, "cn-HelloFont-ID-DianFangSong": 2, "cn-HelloFont-ID-DianHei-CEJ": 3, "cn-HelloFont-ID-DianHei-DEJ": 4, "cn-HelloFont-ID-DianHei-EEJ": 5, "cn-HelloFont-ID-DianHei-FEJ": 6, "cn-HelloFont-ID-DianHei-GEJ": 7, "cn-HelloFont-ID-DianKai-Bold": 8, "cn-HelloFont-ID-DianKai": 9, 
"cn-HelloFont-WenYiHei": 10, "cn-Hellofont-ID-ChenYanXingKai": 11, "cn-Hellofont-ID-DaZiBao": 12, "cn-Hellofont-ID-DaoCaoRen": 13, "cn-Hellofont-ID-JianSong": 14, "cn-Hellofont-ID-JiangHuZhaoPaiHei": 15, "cn-Hellofont-ID-KeSong": 16, "cn-Hellofont-ID-LeYuanTi": 17, "cn-Hellofont-ID-Pinocchio": 18, "cn-Hellofont-ID-QiMiaoTi": 19, 
"cn-Hellofont-ID-QingHuaKai": 20, "cn-Hellofont-ID-QingHuaXingKai": 21, "cn-Hellofont-ID-ShanShuiXingKai": 22, "cn-Hellofont-ID-ShouXieQiShu": 23, "cn-Hellofont-ID-ShouXieTongZhenTi": 24, "cn-Hellofont-ID-TengLingTi": 25, "cn-Hellofont-ID-XiaoLiShu": 26, "cn-Hellofont-ID-XuanZhenSong": 27, "cn-Hellofont-ID-ZhongLingXingKai": 28, "cn-HellofontIDJiaoTangTi": 29, 
"cn-HellofontIDJiuZhuTi": 30, "cn-HuXiaoBao-SaoBao": 31, "cn-HuXiaoBo-NanShen": 32, "cn-HuXiaoBo-ZhenShuai": 33, "cn-SourceHanSansSC-Bold": 34, "cn-SourceHanSansSC-ExtraLight": 35, "cn-SourceHanSansSC-Heavy": 36, "cn-SourceHanSansSC-Light": 37, "cn-SourceHanSansSC-Medium": 38, "cn-SourceHanSansSC-Normal": 39, 
"cn-SourceHanSansSC-Regular": 40, "cn-SourceHanSerifSC-Bold": 41, "cn-SourceHanSerifSC-ExtraLight": 42, "cn-SourceHanSerifSC-Heavy": 43, "cn-SourceHanSerifSC-Light": 44, "cn-SourceHanSerifSC-Medium": 45, "cn-SourceHanSerifSC-Regular": 46, "cn-SourceHanSerifSC-SemiBold": 47, "cn-xiaowei": 48, "cn-AaJianHaoTi": 49, 
"cn-AlibabaPuHuiTi-Bold": 50, "cn-AlibabaPuHuiTi-Heavy": 51, "cn-AlibabaPuHuiTi-Light": 52, "cn-AlibabaPuHuiTi-Medium": 53, "cn-AlibabaPuHuiTi-Regular": 54, "cn-CanvaAcidBoldSC": 55, "cn-CanvaBreezeCN": 56, "cn-CanvaBumperCropSC": 57, "cn-CanvaCakeShopCN": 58, "cn-CanvaEndeavorBlackSC": 59, 
"cn-CanvaJoyHeiCN": 60, "cn-CanvaLiCN": 61, "cn-CanvaOrientalBrushCN": 62, "cn-CanvaPoster": 63, "cn-CanvaQinfuCalligraphyCN": 64, "cn-CanvaSweetHeartCN": 65, "cn-CanvaSwordLikeDreamCN": 66, "cn-CanvaTangyuanHandwritingCN": 67, "cn-CanvaWanderWorldCN": 68, "cn-CanvaWenCN": 69, 
"cn-DianZiChunYi": 70, "cn-GenSekiGothicTW-H": 71, "cn-GenWanMinTW-L": 72, "cn-GenYoMinTW-B": 73, "cn-GenYoMinTW-EL": 74, "cn-GenYoMinTW-H": 75, "cn-GenYoMinTW-M": 76, "cn-GenYoMinTW-R": 77, "cn-GenYoMinTW-SB": 78, "cn-HYQiHei-AZEJ": 79, 
"cn-HYQiHei-EES": 80, "cn-HanaMinA": 81, "cn-HappyZcool-2016": 82, "cn-HelloFont ZJ KeKouKeAiTi": 83, "cn-HelloFont-ID-BoBoTi": 84, "cn-HelloFont-ID-FuGuHei-25": 85, "cn-HelloFont-ID-FuGuHei-35": 86, "cn-HelloFont-ID-FuGuHei-45": 87, "cn-HelloFont-ID-FuGuHei-55": 88, "cn-HelloFont-ID-FuGuHei-65": 89, 
"cn-HelloFont-ID-FuGuHei-75": 90, "cn-HelloFont-ID-FuGuHei-85": 91, "cn-HelloFont-ID-HeiKa": 92, "cn-HelloFont-ID-HeiTang": 93, "cn-HelloFont-ID-JianSong-95": 94, "cn-HelloFont-ID-JueJiangHei-50": 95, "cn-HelloFont-ID-JueJiangHei-55": 96, "cn-HelloFont-ID-JueJiangHei-60": 97, "cn-HelloFont-ID-JueJiangHei-65": 98, "cn-HelloFont-ID-JueJiangHei-70": 99, 
"cn-HelloFont-ID-JueJiangHei-75": 100, "cn-HelloFont-ID-JueJiangHei-80": 101, "cn-HelloFont-ID-KuHeiTi": 102, "cn-HelloFont-ID-LingDongTi": 103, "cn-HelloFont-ID-LingLiTi": 104, "cn-HelloFont-ID-MuFengTi": 105, "cn-HelloFont-ID-NaiNaiJiangTi": 106, "cn-HelloFont-ID-PangDu": 107, "cn-HelloFont-ID-ReLieTi": 108, "cn-HelloFont-ID-RouRun": 109, 
"cn-HelloFont-ID-SaShuangShouXieTi": 110, "cn-HelloFont-ID-WangZheFengFan": 111, "cn-HelloFont-ID-YouQiTi": 112, "cn-Hellofont-ID-XiaLeTi": 113, "cn-Hellofont-ID-XianXiaTi": 114, "cn-HuXiaoBoKuHei": 115, "cn-IDDanMoXingKai": 116, "cn-IDJueJiangHei": 117, "cn-IDMeiLingTi": 118, "cn-IDQQSugar": 119, 
"cn-LiuJianMaoCao-Regular": 120, "cn-LongCang-Regular": 121, "cn-MaShanZheng-Regular": 122, "cn-PangMenZhengDao-3": 123, "cn-PangMenZhengDao-Cu": 124, "cn-PangMenZhengDao": 125, "cn-SentyCaramel": 126, "cn-SourceHanSerifSC": 127, "cn-WenCang-Regular": 128, "cn-WenQuanYiMicroHei": 129, 
"cn-XianErTi": 130, "cn-YRDZSTJF": 131, "cn-YS-HelloFont-BangBangTi": 132, "cn-ZCOOLKuaiLe-Regular": 133, "cn-ZCOOLQingKeHuangYou-Regular": 134, "cn-ZCOOLXiaoWei-Regular": 135, "cn-ZCOOL_KuHei": 136, "cn-ZhiMangXing-Regular": 137, "cn-baotuxiaobaiti": 138, "cn-jiangxizhuokai-Regular": 139, 
"cn-zcool-gdh": 140, "cn-zcoolqingkehuangyouti-Regular": 141, "cn-zcoolwenyiti": 142, "jp-04KanjyukuGothic": 0, "jp-07LightNovelPOP": 1, "jp-07NikumaruFont": 2, "jp-07YasashisaAntique": 3, "jp-07YasashisaGothic": 4, "jp-BokutachinoGothic2Bold": 5, "jp-BokutachinoGothic2Regular": 6, "jp-CHI_SpeedyRight_full_211128-Regular": 7, "jp-CHI_SpeedyRight_italic_full_211127-Regular": 8, "jp-CP-Font": 9, 
"jp-Canva_CezanneProN-B": 10, "jp-Canva_CezanneProN-M": 11, "jp-Canva_ChiaroStd-B": 12, "jp-Canva_CometStd-B": 13, "jp-Canva_DotMincho16Std-M": 14, "jp-Canva_GrecoStd-B": 15, "jp-Canva_GrecoStd-M": 16, "jp-Canva_LyraStd-DB": 17, "jp-Canva_MatisseHatsuhiPro-B": 18, "jp-Canva_MatisseHatsuhiPro-M": 19, 
"jp-Canva_ModeMinAStd-B": 20, "jp-Canva_NewCezanneProN-B": 21, "jp-Canva_NewCezanneProN-M": 22, "jp-Canva_PearlStd-L": 23, "jp-Canva_RaglanStd-UB": 24, "jp-Canva_RailwayStd-B": 25, "jp-Canva_ReggaeStd-B": 26, "jp-Canva_RocknRollStd-DB": 27, "jp-Canva_RodinCattleyaPro-B": 28, "jp-Canva_RodinCattleyaPro-M": 29, 
"jp-Canva_RodinCattleyaPro-UB": 30, "jp-Canva_RodinHimawariPro-B": 31, "jp-Canva_RodinHimawariPro-M": 32, "jp-Canva_RodinMariaPro-B": 33, "jp-Canva_RodinMariaPro-DB": 34, "jp-Canva_RodinProN-M": 35, "jp-Canva_ShadowTLStd-B": 36, "jp-Canva_StickStd-B": 37, "jp-Canva_TsukuAOldMinPr6N-B": 38, "jp-Canva_TsukuAOldMinPr6N-R": 39, 
"jp-Canva_UtrilloPro-DB": 40, "jp-Canva_UtrilloPro-M": 41, "jp-Canva_YurukaStd-UB": 42, "jp-FGUIGEN": 43, "jp-GlowSansJ-Condensed-Heavy": 44, "jp-GlowSansJ-Condensed-Light": 45, "jp-GlowSansJ-Normal-Bold": 46, "jp-GlowSansJ-Normal-Light": 47, "jp-HannariMincho": 48, "jp-HarenosoraMincho": 49, 
"jp-Jiyucho": 50, "jp-Kaiso-Makina-B": 51, "jp-Kaisotai-Next-UP-B": 52, "jp-KokoroMinchoutai": 53, "jp-Mamelon-3-Hi-Regular": 54, "jp-MotoyaAnemoneStd-W1": 55, "jp-MotoyaAnemoneStd-W5": 56, "jp-MotoyaAnticPro-W3": 57, "jp-MotoyaCedarStd-W3": 58, "jp-MotoyaCedarStd-W5": 59, 
"jp-MotoyaGochikaStd-W4": 60, "jp-MotoyaGochikaStd-W8": 61, "jp-MotoyaGothicMiyabiStd-W6": 62, "jp-MotoyaGothicStd-W3": 63, "jp-MotoyaGothicStd-W5": 64, "jp-MotoyaKoinStd-W3": 65, "jp-MotoyaKyotaiStd-W2": 66, "jp-MotoyaKyotaiStd-W4": 67, "jp-MotoyaMaruStd-W3": 68, "jp-MotoyaMaruStd-W5": 69, 
"jp-MotoyaMinchoMiyabiStd-W4": 70, "jp-MotoyaMinchoMiyabiStd-W6": 71, "jp-MotoyaMinchoModernStd-W4": 72, "jp-MotoyaMinchoModernStd-W6": 73, "jp-MotoyaMinchoStd-W3": 74, "jp-MotoyaMinchoStd-W5": 75, "jp-MotoyaReisyoStd-W2": 76, "jp-MotoyaReisyoStd-W6": 77, "jp-MotoyaTohitsuStd-W4": 78, "jp-MotoyaTohitsuStd-W6": 79, 
"jp-MtySousyokuEmBcJis-W6": 80, "jp-MtySousyokuLiBcJis-W6": 81, "jp-Mushin": 82, "jp-NotoSansJP-Bold": 83, "jp-NotoSansJP-Regular": 84, "jp-NudMotoyaAporoStd-W3": 85, "jp-NudMotoyaAporoStd-W5": 86, "jp-NudMotoyaCedarStd-W3": 87, "jp-NudMotoyaCedarStd-W5": 88, "jp-NudMotoyaMaruStd-W3": 89, 
"jp-NudMotoyaMaruStd-W5": 90, "jp-NudMotoyaMinchoStd-W5": 91, "jp-Ounen-mouhitsu": 92, "jp-Ronde-B-Square": 93, "jp-SMotoyaGyosyoStd-W5": 94, "jp-SMotoyaSinkaiStd-W3": 95, "jp-SMotoyaSinkaiStd-W5": 96, "jp-SourceHanSansJP-Bold": 97, "jp-SourceHanSansJP-Regular": 98, "jp-SourceHanSerifJP-Bold": 99, 
"jp-SourceHanSerifJP-Regular": 100, "jp-TazuganeGothicStdN-Bold": 101, "jp-TazuganeGothicStdN-Regular": 102, "jp-TelopMinProN-B": 103, "jp-Togalite-Bold": 104, "jp-Togalite-Regular": 105, "jp-TsukuMinPr6N-E": 106, "jp-TsukuMinPr6N-M": 107, "jp-mikachan_o": 108, "jp-nagayama_kai": 109, 
"jp-07LogoTypeGothic7": 110, "jp-07TetsubinGothic": 111, "jp-851CHIKARA-DZUYOKU-KANA-A": 112, "jp-ARMinchoJIS-Light": 113, "jp-ARMinchoJIS-Ultra": 114, "jp-ARPCrystalMinchoJIS-Medium": 115, "jp-ARPCrystalRGothicJIS-Medium": 116, "jp-ARShounanShinpitsuGyosyoJIS-Medium": 117, "jp-AozoraMincho-bold": 118, "jp-AozoraMinchoRegular": 119, 
"jp-ArialUnicodeMS-Bold": 120, "jp-ArialUnicodeMS": 121, "jp-CanvaBreezeJP": 122, "jp-CanvaLiCN": 123, "jp-CanvaLiJP": 124, "jp-CanvaOrientalBrushCN": 125, "jp-CanvaQinfuCalligraphyJP": 126, "jp-CanvaSweetHeartJP": 127, "jp-CanvaWenJP": 128, "jp-Corporate-Logo-Bold": 129, 
"jp-DelaGothicOne-Regular": 130, "jp-GN-Kin-iro_SansSerif": 131, "jp-GN-Koharuiro_Sunray": 132, "jp-GenEiGothicM-B": 133, "jp-GenEiGothicM-R": 134, "jp-GenJyuuGothic-Bold": 135, "jp-GenRyuMinTW-B": 136, "jp-GenRyuMinTW-R": 137, "jp-GenSekiGothicTW-B": 138, "jp-GenSekiGothicTW-R": 139, 
"jp-GenSenRoundedTW-B": 140, "jp-GenSenRoundedTW-R": 141, "jp-GenShinGothic-Bold": 142, "jp-GenShinGothic-Normal": 143, "jp-GenWanMinTW-L": 144, "jp-GenYoGothicTW-B": 145, "jp-GenYoGothicTW-R": 146, "jp-GenYoMinTW-B": 147, "jp-GenYoMinTW-R": 148, "jp-HGBouquet": 149, 
"jp-HanaMinA": 150, "jp-HanazomeFont": 151, "jp-HinaMincho-Regular": 152, "jp-Honoka-Antique-Maru": 153, "jp-Honoka-Mincho": 154, "jp-HuiFontP": 155, "jp-IPAexMincho": 156, "jp-JK-Gothic-L": 157, "jp-JK-Gothic-M": 158, "jp-JackeyFont": 159, 
"jp-KaiseiTokumin-Bold": 160, "jp-KaiseiTokumin-Regular": 161, "jp-Keifont": 162, "jp-KiwiMaru-Regular": 163, "jp-Koku-Mincho-Regular": 164, "jp-MotoyaLMaru-W3-90ms-RKSJ-H": 165, "jp-NewTegomin-Regular": 166, "jp-NicoKaku": 167, "jp-NicoMoji+": 168, "jp-Otsutome_font-Bold": 169, 
"jp-PottaOne-Regular": 170, "jp-RampartOne-Regular": 171, "jp-Senobi-Gothic-Bold": 172, "jp-Senobi-Gothic-Regular": 173, "jp-SmartFontUI-Proportional": 174, "jp-SoukouMincho": 175, "jp-TEST_Klee-DB": 176, "jp-TEST_Klee-M": 177, "jp-TEST_UDMincho-B": 178, "jp-TEST_UDMincho-L": 179, 
"jp-TT_Akakane-EB": 180, "jp-Tanuki-Permanent-Marker": 181, "jp-TrainOne-Regular": 182, "jp-TsunagiGothic-Black": 183, "jp-Ume-Hy-Gothic": 184, "jp-Ume-P-Mincho": 185, "jp-WenQuanYiMicroHei": 186, "jp-XANO-mincho-U32": 187, "jp-YOzFontM90-Regular": 188, "jp-Yomogi-Regular": 189, 
"jp-YujiBoku-Regular": 190, "jp-YujiSyuku-Regular": 191, "jp-ZenKakuGothicNew-Bold": 192, "jp-ZenKakuGothicNew-Regular": 193, "jp-ZenKurenaido-Regular": 194, "jp-ZenMaruGothic-Bold": 195, "jp-ZenMaruGothic-Regular": 196, "jp-darts-font": 197, "jp-irohakakuC-Bold": 198, "jp-irohakakuC-Medium": 199, 
"jp-irohakakuC-Regular": 200, "jp-katyou": 201, "jp-mplus-1m-bold": 202, "jp-mplus-1m-regular": 203, "jp-mplus-1p-bold": 204, "jp-mplus-1p-regular": 205, "jp-rounded-mplus-1p-bold": 206, "jp-rounded-mplus-1p-regular": 207, "jp-timemachine-wa": 208, "jp-ttf-GenEiLateMin-Medium": 209, 
"jp-uzura_font": 210, "kr-Arita-buri-Bold_OTF": 0, "kr-Arita-buri-HairLine_OTF": 1, "kr-Arita-buri-Light_OTF": 2, "kr-Arita-buri-Medium_OTF": 3, "kr-Arita-buri-SemiBold_OTF": 4, "kr-Canva_YDSunshineL": 5, "kr-Canva_YDSunshineM": 6, "kr-Canva_YoonGulimPro710": 7, "kr-Canva_YoonGulimPro730": 8, "kr-Canva_YoonGulimPro740": 9, 
"kr-Canva_YoonGulimPro760": 10, "kr-Canva_YoonGulimPro770": 11, "kr-Canva_YoonGulimPro790": 12, "kr-CreHappB": 13, "kr-CreHappL": 14, "kr-CreHappM": 15, "kr-CreHappS": 16, "kr-OTAuroraB": 17, "kr-OTAuroraL": 18, "kr-OTAuroraR": 19, 
"kr-OTDoldamgilB": 20, "kr-OTDoldamgilL": 21, "kr-OTDoldamgilR": 22, "kr-OTHamsterB": 23, "kr-OTHamsterL": 24, "kr-OTHamsterR": 25, "kr-OTHapchangdanB": 26, "kr-OTHapchangdanL": 27, "kr-OTHapchangdanR": 28, "kr-OTSupersizeBkBOX": 29, 
"kr-SourceHanSansKR-Bold": 30, "kr-SourceHanSansKR-ExtraLight": 31, "kr-SourceHanSansKR-Heavy": 32, "kr-SourceHanSansKR-Light": 33, "kr-SourceHanSansKR-Medium": 34, "kr-SourceHanSansKR-Normal": 35, "kr-SourceHanSansKR-Regular": 36, "kr-SourceHanSansSC-Bold": 37, "kr-SourceHanSansSC-ExtraLight": 38, "kr-SourceHanSansSC-Heavy": 39, 
"kr-SourceHanSansSC-Light": 40, "kr-SourceHanSansSC-Medium": 41, "kr-SourceHanSansSC-Normal": 42, "kr-SourceHanSansSC-Regular": 43, "kr-SourceHanSerifSC-Bold": 44, "kr-SourceHanSerifSC-SemiBold": 45, "kr-TDTDBubbleBubbleOTF": 46, "kr-TDTDConfusionOTF": 47, "kr-TDTDCuteAndCuteOTF": 48, "kr-TDTDEggTakOTF": 49, 
"kr-TDTDEmotionalLetterOTF": 50, "kr-TDTDGalapagosOTF": 51, "kr-TDTDHappyHourOTF": 52, "kr-TDTDLatteOTF": 53, "kr-TDTDMoonLightOTF": 54, "kr-TDTDParkForestOTF": 55, "kr-TDTDPencilOTF": 56, "kr-TDTDSmileOTF": 57, "kr-TDTDSproutOTF": 58, "kr-TDTDSunshineOTF": 59, 
"kr-TDTDWaferOTF": 60, "kr-777Chyaochyureu": 61, "kr-ArialUnicodeMS-Bold": 62, "kr-ArialUnicodeMS": 63, "kr-BMHANNA": 64, "kr-Baekmuk-Dotum": 65, "kr-BagelFatOne-Regular": 66, "kr-CoreBandi": 67, "kr-CoreBandiFace": 68, "kr-CoreBori": 69, 
"kr-DoHyeon-Regular": 70, "kr-Dokdo-Regular": 71, "kr-Gaegu-Bold": 72, "kr-Gaegu-Light": 73, "kr-Gaegu-Regular": 74, "kr-GamjaFlower-Regular": 75, "kr-GasoekOne-Regular": 76, "kr-GothicA1-Black": 77, "kr-GothicA1-Bold": 78, "kr-GothicA1-ExtraBold": 79, 
"kr-GothicA1-ExtraLight": 80, "kr-GothicA1-Light": 81, "kr-GothicA1-Medium": 82, "kr-GothicA1-Regular": 83, "kr-GothicA1-SemiBold": 84, "kr-GothicA1-Thin": 85, "kr-Gugi-Regular": 86, "kr-HiMelody-Regular": 87, "kr-Jua-Regular": 88, "kr-KirangHaerang-Regular": 89, 
"kr-NanumBrush": 90, "kr-NanumPen": 91, "kr-NanumSquareRoundB": 92, "kr-NanumSquareRoundEB": 93, "kr-NanumSquareRoundL": 94, "kr-NanumSquareRoundR": 95, "kr-SeH-CB": 96, "kr-SeH-CBL": 97, "kr-SeH-CEB": 98, "kr-SeH-CL": 99, 
"kr-SeH-CM": 100, "kr-SeN-CB": 101, "kr-SeN-CBL": 102, "kr-SeN-CEB": 103, "kr-SeN-CL": 104, "kr-SeN-CM": 105, "kr-Sunflower-Bold": 106, "kr-Sunflower-Light": 107, "kr-Sunflower-Medium": 108, "kr-TTClaytoyR": 109, 
"kr-TTDalpangiR": 110, "kr-TTMamablockR": 111, "kr-TTNauidongmuR": 112, "kr-TTOktapbangR": 113, "kr-UhBeeMiMi": 114, "kr-UhBeeMiMiBold": 115, "kr-UhBeeSe_hyun": 116, "kr-UhBeeSe_hyunBold": 117, "kr-UhBeenamsoyoung": 118, "kr-UhBeenamsoyoungBold": 119, 
"kr-WenQuanYiMicroHei": 120, "kr-YeonSung-Regular": 121}"""


def add_special_token(tokenizer: T5Tokenizer, text_encoder: T5Stack):
    """
    Add special tokens for color and font to tokenizer and text encoder.

    Args:
        tokenizer: Huggingface tokenizer.
        text_encoder: Huggingface T5 encoder.
    """
    idx_font_dict = json.loads(MULTILINGUAL_10_LANG_IDX_JSON)
    idx_color_dict = json.loads(COLOR_IDX_JSON)

    font_token = [f"<{font_code[:2]}-font-{idx_font_dict[font_code]}>" for font_code in idx_font_dict]
    color_token = [f"<color-{i}>" for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    additional_special_tokens += color_token
    additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # Set mean_resizing=False to avoid PyTorch LAPACK dependency
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> Tuple[T5Stack, T5Tokenizer]:
    BYT5_CONFIG_JSON = """
{
    "_name_or_path": "/home/patrick/t5/byt5-small",
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 3584,
    "d_kv": 64,
    "d_model": 1472,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "gradient_checkpointing": false,
    "initializer_factor": 1.0,
    "is_encoder_decoder": true,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 4,
    "num_heads": 6,
    "num_layers": 12,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": false,
    "tokenizer_class": "ByT5Tokenizer",
    "transformers_version": "4.7.0.dev0",
    "use_cache": true,
    "vocab_size": 384
    }
"""

    logger.info(f"Loading BYT5 tokenizer from {BYT5_TOKENIZER_PATH}")
    byt5_tokenizer = AutoTokenizer.from_pretrained(BYT5_TOKENIZER_PATH)

    logger.info("Initializing BYT5 text encoder")
    config = json.loads(BYT5_CONFIG_JSON)
    config = T5Config(**config)
    with init_empty_weights():
        byt5_text_encoder = T5ForConditionalGeneration._from_config(config).get_encoder()

    add_special_token(byt5_tokenizer, byt5_text_encoder)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device, disable_mmap=disable_mmap, dtype=dtype)

    # remove "encoder." prefix
    sd = {k[len("encoder.") :] if k.startswith("encoder.") else k: v for k, v in sd.items()}
    sd["embed_tokens.weight"] = sd.pop("shared.weight")

    info = byt5_text_encoder.load_state_dict(sd, strict=True, assign=True)
    byt5_text_encoder.to(device)
    byt5_text_encoder.eval()
    logger.info(f"BYT5 text encoder loaded with info: {info}")

    return byt5_tokenizer, byt5_text_encoder


def load_qwen2_5_vl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration]:
    QWEN2_5_VL_CONFIG_JSON = """
{
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 128000,
  "max_window_layers": 28,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "mrope_section": [
      16,
      24,
      24
    ],
    "rope_type": "default",
    "type": "default"
  },
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "text_config": {
    "architectures": [
      "Qwen2_5_VLForConditionalGeneration"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "image_token_id": null,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "layer_types": [
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention"
    ],
    "max_position_embeddings": 128000,
    "max_window_layers": 28,
    "model_type": "qwen2_5_vl_text",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_section": [
        16,
        24,
        24
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "torch_dtype": "float32",
    "use_cache": true,
    "use_sliding_window": false,
    "video_token_id": null,
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "vision_token_id": 151654,
    "vocab_size": 152064
  },
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.1",
  "use_cache": true,
  "use_sliding_window": false,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "hidden_act": "silu",
    "hidden_size": 1280,
    "in_channels": 3,
    "in_chans": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3420,
    "model_type": "qwen2_5_vl",
    "num_heads": 16,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "tokens_per_second": 2,
    "torch_dtype": "float32",
    "window_size": 112
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 152064
}
"""
    config = json.loads(QWEN2_5_VL_CONFIG_JSON)
    config = Qwen2_5_VLConfig(**config)
    with init_empty_weights():
        qwen2_5_vl = Qwen2_5_VLForConditionalGeneration._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device, disable_mmap=disable_mmap, dtype=dtype)

    # convert prefixes
    for key in list(sd.keys()):
        if key.startswith("model."):
            new_key = key.replace("model.", "model.language_model.", 1)
        elif key.startswith("visual."):
            new_key = key.replace("visual.", "model.visual.", 1)
        else:
            continue
        if key not in sd:
            logger.warning(f"Key {key} not found in state dict, skipping.")
            continue
        sd[new_key] = sd.pop(key)

    info = qwen2_5_vl.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen2.5-VL: {info}")
    qwen2_5_vl.to(device)
    qwen2_5_vl.eval()

    if dtype is not None:
        if dtype.itemsize == 1:  # fp8
            org_dtype = torch.bfloat16  # model weight is fp8 in loading, but original dtype is bfloat16
            logger.info(f"prepare Qwen2.5-VL for fp8: set to {dtype} from {org_dtype}")
            qwen2_5_vl.to(dtype)

            # prepare LLM for fp8
            def prepare_fp8(vl_model: Qwen2_5_VLForConditionalGeneration, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        input_dtype = hidden_states.dtype
                        hidden_states = hidden_states.to(torch.float32)
                        variance = hidden_states.pow(2).mean(-1, keepdim=True)
                        hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                        # return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
                        return (module.weight.to(torch.float32) * hidden_states.to(torch.float32)).to(input_dtype)

                    return forward

                def decoder_forward_hook(module):
                    def forward(
                        hidden_states: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_value: Optional[tuple[torch.Tensor]] = None,
                        output_attentions: Optional[bool] = False,
                        use_cache: Optional[bool] = False,
                        cache_position: Optional[torch.LongTensor] = None,
                        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                        **kwargs,
                    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

                        residual = hidden_states

                        hidden_states = module.input_layernorm(hidden_states)

                        # Self Attention
                        hidden_states, self_attn_weights = module.self_attn(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                        input_dtype = hidden_states.dtype
                        hidden_states = residual.to(torch.float32) + hidden_states.to(torch.float32)
                        hidden_states = hidden_states.to(input_dtype)

                        # Fully Connected
                        residual = hidden_states
                        hidden_states = module.post_attention_layernorm(hidden_states)
                        hidden_states = module.mlp(hidden_states)
                        hidden_states = residual + hidden_states

                        outputs = (hidden_states,)

                        if output_attentions:
                            outputs += (self_attn_weights,)

                        return outputs

                    return forward

                for module in vl_model.modules():
                    if module.__class__.__name__ in ["Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["Qwen2RMSNorm"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)
                    if module.__class__.__name__ in ["Qwen2_5_VLDecoderLayer"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = decoder_forward_hook(module)
                    if module.__class__.__name__ in ["Qwen2_5_VisionRotaryEmbedding"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.to(target_dtype)

            prepare_fp8(qwen2_5_vl, org_dtype)

        else:
            logger.info(f"Setting Qwen2.5-VL to dtype: {dtype}")
            qwen2_5_vl.to(dtype)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {QWEN_2_5_VL_IMAGE_ID}")
    tokenizer = Qwen2Tokenizer.from_pretrained(QWEN_2_5_VL_IMAGE_ID)
    return tokenizer, qwen2_5_vl


TOKENIZER_MAX_LENGTH = 1024
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


def get_qwen_prompt_embeds(
    tokenizer: Qwen2Tokenizer, vlm: Qwen2_5_VLForConditionalGeneration, prompt: Union[str, list[str]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, mask = get_qwen_tokens(tokenizer, prompt)
    return get_qwen_prompt_embeds_from_tokens(vlm, input_ids, mask)


def get_qwen_tokens(tokenizer: Qwen2Tokenizer, prompt: Union[str, list[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer_max_length = TOKENIZER_MAX_LENGTH

    # HunyuanImage-2.1 does not use "<|im_start|>assistant\n" in the prompt template
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
    # \n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
    # default_sample_size = 128

    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt")
    return txt_tokens.input_ids, txt_tokens.attention_mask


def get_qwen_prompt_embeds_from_tokens(
    vlm: Qwen2_5_VLForConditionalGeneration, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer_max_length = TOKENIZER_MAX_LENGTH
    drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX

    device = vlm.device
    dtype = vlm.dtype

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)

    if dtype.itemsize == 1:  # fp8
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            encoder_hidden_states = vlm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    else:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=True):
            encoder_hidden_states = vlm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = encoder_hidden_states.hidden_states[-3]  # use the 3rd last layer's hidden states for HunyuanImage-2.1
    if hidden_states.shape[1] > tokenizer_max_length + drop_idx:
        logger.warning(f"Hidden states shape {hidden_states.shape} exceeds max length {tokenizer_max_length + drop_idx}")

    # --- Unnecessary complicated processing, keep for reference ---
    # split_hidden_states = extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    # split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    # attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    # max_seq_len = max([e.size(0) for e in split_hidden_states])
    # prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
    # encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
    # ----------------------------------------------------------

    prompt_embeds = hidden_states[:, drop_idx:, :]
    encoder_attention_mask = attention_mask[:, drop_idx:]
    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds, encoder_attention_mask


def format_prompt(texts, styles):
    """
    Text "{text}" in {color}, {type}.
    """

    prompt = ""
    for text, style in zip(texts, styles):
        # color and style are always None in official implementation, so we only use text
        text_prompt = f'Text "{text}"'
        text_prompt += ". "
        prompt = prompt + text_prompt
    return prompt


BYT5_MAX_LENGTH = 128


def get_glyph_prompt_embeds(
    tokenizer: T5Tokenizer, text_encoder: T5Stack, prompt: Optional[str] = None
) -> Tuple[list[bool], torch.Tensor, torch.Tensor]:
    byt5_tokens, byt5_text_mask = get_byt5_text_tokens(tokenizer, prompt)
    return get_byt5_prompt_embeds_from_tokens(text_encoder, byt5_tokens, byt5_text_mask)


def get_byt5_prompt_embeds_from_tokens(
    text_encoder: T5Stack, byt5_text_ids: Optional[torch.Tensor], byt5_text_mask: Optional[torch.Tensor]
) -> Tuple[list[bool], torch.Tensor, torch.Tensor]:
    byt5_max_length = BYT5_MAX_LENGTH

    if byt5_text_ids is None or byt5_text_mask is None or byt5_text_mask.sum() == 0:
        return (
            [False],
            torch.zeros((1, byt5_max_length, 1472), device=text_encoder.device),
            torch.zeros((1, byt5_max_length), device=text_encoder.device, dtype=torch.int64),
        )

    byt5_text_ids = byt5_text_ids.to(device=text_encoder.device)
    byt5_text_mask = byt5_text_mask.to(device=text_encoder.device)

    with torch.no_grad(), torch.autocast(device_type=text_encoder.device.type, dtype=text_encoder.dtype, enabled=True):
        byt5_prompt_embeds = text_encoder(byt5_text_ids, attention_mask=byt5_text_mask.float())
    byt5_emb = byt5_prompt_embeds[0]

    return [True], byt5_emb, byt5_text_mask


def get_byt5_text_tokens(tokenizer, prompt):
    if not prompt:
        return None, None

    try:
        text_prompt_texts = []
        # pattern_quote_single = r"\'(.*?)\'"
        pattern_quote_double = r"\"(.*?)\""
        pattern_quote_chinese_single = r"‘(.*?)’"
        pattern_quote_chinese_double = r"“(.*?)”"

        # matches_quote_single = re.findall(pattern_quote_single, prompt)
        matches_quote_double = re.findall(pattern_quote_double, prompt)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

        # text_prompt_texts.extend(matches_quote_single)
        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        if not text_prompt_texts:
            return None, None

        text_prompt_style_list = [{"color": None, "font-family": None} for _ in range(len(text_prompt_texts))]
        glyph_text_formatted = format_prompt(text_prompt_texts, text_prompt_style_list)
        logger.info(f"Glyph text formatted: {glyph_text_formatted}")

        byt5_text_inputs = tokenizer(
            glyph_text_formatted,
            padding="max_length",
            max_length=BYT5_MAX_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    except Exception as e:
        logger.warning(f"Warning: Error in glyph encoding, using fallback: {e}")
        return None, None
