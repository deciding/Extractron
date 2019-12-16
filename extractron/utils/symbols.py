'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

_pad = '_'
_eos = '~'
_pinyinchars = 'abcdefghijklmnopqrstuvwxyz1234567890'
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? '
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!\'(),-.:;? #$%^'
_english2latin = 'ƖƗƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺƻƼƽƾƿǂǄǅǆǇǈǉǊǋǌǍ'
# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + list(_english2latin)
CMUPhonemes=['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B',  'CH', 'D',  'DH', 'EH', 'ER', 'EY', 'F',  'G',  'HH', 'IH', 'IY', 'JH', 'K',  'L',  'M',  'N',  'NG', 'OW', 'OY', 'P',  'R',  'S',  'SH', 'T',  'TH', 'UH', 'UW', 'V',  'W',  'Y',  'Z',  'ZH', '0', '1', '2', '3']
puncs = '!\'(),-.:;? '
CMUPhonemes += list(puncs)
CMUPhonemes = [_pad, _eos] + CMUPhonemes
