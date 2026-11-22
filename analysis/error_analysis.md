# Error Analysis — `C4a_fastconformer-usc-ft-rnnt` (USC Test)

- **Evaluated model:** `C4a_fastconformer-usc-ft-rnnt` (test WER: 11.38%)
- **Total test clips:** 3836 | **Erroneous clips:** 1783 (46.5%)
- **Test OOV rate (token-based):** 5823 / 27042 (**21.53%**)
- **Test OOV rate (unique word/type-based):** 4773 / 10572 (**45.15%**)

## Overall Error Distribution (Entire Test Set)

| Error Type | Count | Share of Total Errors |
|---|---:|---:|
| Substitution | 2464 | 80.1% |
| Deletion | 445 | 14.5% |
| Insertion | 169 | 5.5% |
| **Total** | **3078** | **100.0%** |

## Substitution Breakdown (Automatic Rule)

Automatic rule: word pairs that become identical after removing the apostrophe (`'`) are classified as **apostrophe errors**, pairs with a common prefix of at least 3 characters and at least 50% of the shorter word are classified as **suffix errors**, and all others are classified as **root/other errors**.

**These categories are an automatic estimate, not a manual classification.** The rule only compares the first letters of the two words, so it mislabels some pairs, and the sample of erroneous sentences below was not reviewed by hand:

- 25 substitutions counted as suffix errors differ from the reference only by a doubled letter, e.g. `elchinning` → `elchining` (7 times; *Elchin*, a name, vs *elchi* "ambassador") and `hurkkan` → `hurkan`. Many of these are spelling slips or different stems rather than wrong suffixes, so the suffix share is an upper estimate.
- Spelling variants with a dropped initial *y* (`etarli` → `yetarli`, `etkazadigan` → `yetkazadigan`) are counted as root/other errors; the rule has no spelling category.
- Some references are themselves wrong, and the model output is counted as an error anyway. In clip `1187023182_2_35103_1` the reference reads *beshinchi davlatda aylandi* and the model writes *beshinchi davlatga aylandi*, the grammatical form ("became the fifth country").

| Substitution Subtype | Count | Share of Substitutions |
|---|---:|---:|
| Suffix error (common prefix) | 1065 | 43.2% |
| Root/other substitution error | 1334 | 54.1% |
| Apostrophe error (`'`) | 65 | 2.6% |

### Most Frequent Suffix Errors (Top 10)

| Reference | Prediction | Count |
|---|---|---:|
| `elchinning` | `elchining` | 7 |
| `abduzafar` | `abdusar` | 5 |
| `o'ng` | `o'n` | 3 |
| `eshigi` | `eshik` | 3 |
| `paytlar` | `paytda` | 2 |
| `olishi` | `olish` | 2 |
| `sening` | `sen` | 2 |
| `ko'rsatuvi` | `ko'rsatuv` | 2 |
| `otarchi` | `otachi` | 2 |
| `elchinni` | `elchini` | 2 |

### Most Frequent Apostrophe Errors (Top 10)

| Reference | Prediction | Count |
|---|---|---:|
| `no'yon` | `noyon` | 5 |
| `ya'ni` | `yani` | 3 |
| `mo''jiza` | `mo'jiza` | 2 |
| `qat'iyatliroq` | `qatiyatliroq` | 2 |
| `she'rda` | `sherda` | 2 |
| `bo'limiga` | `bolimiga` | 1 |
| `instagram'da` | `instagramda` | 1 |
| `amazon'ning` | `amazonning` | 1 |
| `qur'on` | `quron` | 1 |
| `lego'da` | `leg'oda` | 1 |

## Random Sample of 100 Erroneous Test Sentences (`seed=42`)

| # | Clip ID | Reference (normalized) | Prediction (normalized) | Error Categories |
|---:|---|---|---|---|
| 1 | `709790549_1_21264_1` | ertami kechmi hamma hamma uchun bo'ladi agar vaqt etarli bo'lsa | ertami kechmi hamma uchun bo'ladi agar vaqt yetarli bo'lsa | deletion(1), root_or_other(1) |
| 2 | `1187023182_2_35519_1` | ilgari bu yerni faqat suratlarda ko'rgandim va qandaydir sehrli joy ekanini bilardim | ilgari bu yerni faqat suratlarda ko'rgandim va qandaydir sehrli choy ekanini bilardim | root_or_other(1) |
| 3 | `1187023182_2_35103_1` | buyuk britaniya koronavirus aniqlangan yevropadagi beshinchi davlatda aylandi | buyuk britaniya koronavirus aniqlangan yevropadagi beshinchi davlatga aylandi | suffix_error(1) |
| 4 | `909734087_2_29937_1` | chin shodlik shishalar bo'shatilguncha ekan | chin shodlik shishalar bo'shatilgancha ekan | suffix_error(1) |
| 5 | `1187023182_2_38066_1` | anvar xotinini quchib erkalagan bo'ldi | anvar xotinni quchib erkalaganday bo'ldi | suffix_error(2) |
| 6 | `1187023182_2_37329_1` | sovchilar uy sohibasining ko'nglini ko'taradigan bir ikki gaplarni aytib lutf ko'rgazib o'rinlaridan jilishdi | sovchilar uy sohibasining ko'nglini ko'taradigan bir ikki gaplarni aytib lutf ko'rgazib o'rnilaridan tilishdi | root_or_other(2) |
| 7 | `1187023182_2_37215_1` | zelixon vaziyatni yumshatish uchun gapni hazilga burdi | silxon vaziyatni yumshatish uchun gapni hazilga yugurdi | root_or_other(2) |
| 8 | `1187023182_2_36073_1` | qarang qu yaqinga tengmi yoki teng dani la bilan almashtirsa bo'ladimi | qarang ku yaqinida dengmi yoki teng dala bilan almashtirsam bo'ladimi | root_or_other(3), suffix_error(2), deletion(1) |
| 9 | `909734087_2_29918_1` | shu ahvolda ekanida akasining ihraganini eshitdi | shu ahvolda ekan akasining ingraganini eshitdi | suffix_error(1), root_or_other(1) |
| 10 | `1187023182_2_35473_1` | ismli erkak og'ir tan jahorati olib shifoxonada vafot etgan | ismli erkak og'ir tan jarohati olib shifoxonada vafot etgan | root_or_other(1) |
| 11 | `818938633_2_82663_1` | bo'lganlar haqiqatan ham g'ayrioddiy fazilatlarga ega | lar haqiqatan ham g'ayrioddiy fazilatlarga ega | root_or_other(1) |
| 12 | `909734087_2_29934_1` | hay o'rtoqlaringga qo'shilib ichmagin dedi | hay o'rtoqlarimga qo'shilib ichmang dedi | suffix_error(2) |
| 13 | `1571110404_2_36665_1` | jingalaksoch chi | jingalak sochchi | suffix_error(1), root_or_other(1) |
| 14 | `1187023182_2_35407_1` | grand road yetakchi va ishonchli hamkor | grand rout yetakchi va ishonchli hamkor | root_or_other(1) |
| 15 | `1571110404_2_37688_1` | aslida bu risolat kampir bolalarining kimga tortganini yaxshi biladi | aslida bu risolat kampir bolalarining kimga tortganligini yaxshi biladi | suffix_error(1) |
| 16 | `1289864764_2_32803_2` | endi bu masalaga ham butunlay yangicha yondashib mas'uliyat | endi bu masalaga ham butunlay yangicha yondashmas mas'uliyat | suffix_error(1) |
| 17 | `1187023182_2_35133_1` | daryo dam olish kunlari respublikasi turkiston hududidan juda iliq va quruq havo oqimi kiradi | daryo dalish kunlari respublika turkiston hududidan juda iliq va quruq havo oqimi kiradi | deletion(1), root_or_other(1), suffix_error(1) |
| 18 | `1187023182_2_35121_1` | bemorlar xorijga sayohat qilgan mahalliy aholi vakillari yoki mamlakatga kelgan sayyohlar ekanligiga aniqlik kiritilmpqda | bemalol xorijga sayohat qilgan mahalliy aholi vakillari va mahalla mamlakatdan kelgan sayyohlar ekanligiga aniqlik kiritilmoqda | root_or_other(2), suffix_error(2), insertion(1) |
| 19 | `1187023182_2_35434_1` | seni muxlisingman suratga tushsak bo'ladimi | seni muxlisingman suratga tushirsak bo'ladimi | suffix_error(1) |
| 20 | `1187023182_2_37193_1` | asadbek shu gapi bilan suhbatga yakun yasadi | asadbek shu kabi suhbatga yakun yasadi | deletion(1), root_or_other(1) |
| 21 | `1187023182_2_37262_1` | qiz tipirchilab yotganida bu o'git esga kelarmidi | qiz tipirchilab yotganda bu o'g'il esga kelarmidi | suffix_error(2) |
| 22 | `1571110404_2_36457_1` | zinatulinning ishi shoshqich emas shu bois ikkinchi hukm unga nasib etdi | zinadulning ishi shoshqich emas shu bois ikkinchi hukm unga nasib etdi | root_or_other(1) |
| 23 | `265773029_2_33374_2` | avvalroq xitoyda tahsil olayotgan o'zbekistonlik talabalarning bir guruhi o'zbekistonga qaytarilgani haqida xabar berilgan edi | avvalroq avvalroq xitoyda tahsil olayotgan o'zbekistonlik talabalarning bir guruhi o'zbekistonga qaytarilgani haqida xabar berilgan edi | insertion(1) |
| 24 | `1187023182_2_35106_1` | virus baykalorti o'lkasi va tyumen viloyatida bo'lgan xitoyning ikki fuqarosida aniqlangan | virus bayalog o'lkasi va tyumen viloyatida bo'lgan xitoyning ikki fuqarosida aniqlangan | root_or_other(1) |
| 25 | `1571110404_2_36738_1` | qirq yettinchi statya bilan bo'shatilsin ha | qirq yettinchi statsiya bilan bo'shatilsin ha | suffix_error(1) |
| 26 | `1187023182_2_37112_1` | ha shunaqaroq dedi elchin asadbekning pichinglarini o'zicha talqin qilib | ha shunaqaroq dedi elchin asadbekning pichanlarini o'zicha talqin qilib | root_or_other(1) |
| 27 | `909734087_2_29822_1` | uyg'ondingmi toychoq dedi dadasi ostonada turib | yovdingmi toychoq dedi dadasi ostovada turib | root_or_other(1), suffix_error(1) |
| 28 | `807286871_2_94292_1` | kengashining navbatdan tashqari sessiyasi | kengashining navbatdan tashqari ses | suffix_error(1) |
| 29 | `909734087_2_29085_1` | shu vahima gaplar kuchaya borgach tekshirib ko'rish zohidga topshirildi | shu vahima gaplar kuchayib borgach tekshirib ko'rish zohidga topshirildi | suffix_error(1) |
| 30 | `1289864764_2_32769_2` | toshkent shahri mustaqil davlatimizning poytaxti ta'bir joiz bo'lsa o'zbekistonimizning yuragi o'zbekistonimizning yuzidir | toshkent shahri mustaqil davlati bugun poytaxti yuqori va o'zbekistonimizning yuragi o'zbekistonimizning yuzidir | root_or_other(4), suffix_error(1) |
| 31 | `1187023182_2_37206_1` | shunday zamonda shunday qudrat sohibi bo'lmish asadbekning qadam bosishi ularni hayrat to'rlarida o'rab tashladi | shunday zamonda shunday qudrat sohibi bo'lmish asadbekning qadam bosishi ularni hayrat nurlarida o'rab tashladi | root_or_other(1) |
| 32 | `1571110404_2_19267_1` | yoy o'qi nishon yog'ochiga taq etib urilgani baralla eshitildi | yoy o'qinshon yog'ochiga taqit urilgani baralla eshitildi | deletion(2), root_or_other(2) |
| 33 | `1571110404_2_36868_1` | xotini gapiga amal qilib uch marta astag'firulloh deb qo'yadi | xotini gapiga amal qilib uch marta astaloh deb qo'yadi | suffix_error(1) |
| 34 | `1187023182_2_38079_1` | murdalar to'zigan xazon singari asadbek atrofida charx urarmish | mo'rtalar to'zidan g'azon singari asadbek atrofida charx urarmish | root_or_other(2), suffix_error(1) |
| 35 | `909734087_2_82522_1` | shu maqsadda bir | shu maqsadda | deletion(1) |
| 36 | `972644779_2_87532_3` | quyidagilari bilan diqqatga sazovor | uydagilari bilan diqqatga sazovor | root_or_other(1) |
| 37 | `1187023182_2_34579_1` | clubhouse qanday ijtimoiy tarmoq | kupxaus qanday ijtimoiy tarmoq | root_or_other(1) |
| 38 | `909734087_2_30046_1` | hecham unaqamas | unaqamas | deletion(1) |
| 39 | `909734087_2_81812_1` | xalq og'zidagi mish | xalq og'zidagi mishmish | suffix_error(1) |
| 40 | `1187023182_2_36925_1` | zeli og'a bu o'zim bilan yerga kiradigan sir xafa bo'lmang | zelo bu o'zim bilan yerga kiradigan sir xafa bo'lmang | deletion(1), root_or_other(1) |
| 41 | `909734087_2_29065_1` | buni ko'chadagi mishiqi bola ham biladi sen bilmaysanmi qishloqi zohid ham eshitgan bu gaplarni | buni ko'chadagi mishki bola ham biladi sen bilmaysanmi qishloqi zohid ham eshitgan bu gaplarni | suffix_error(1) |
| 42 | `1289864764_2_32910_1` | taraqqiyotimizning o'sish sur'atlari tobora oshib boryapti | ning o'sish sur'atlari tobora oshib boryapti | root_or_other(1) |
| 43 | `1187023182_2_38327_1` | ayollarning kalta paltosini kiyib olgan novcha odam bir haykalcha ro'parasida turib olib nuqul kulardi | ayollarning paltosini kiyib olgan nocha odam bir haykalcha ro'parasida turib olib nuqul kulardi | deletion(1), root_or_other(1) |
| 44 | `1187023182_2_36901_1` | baxtiga zelixon bor ekan yolg'izlatmadi | baxtiga zeli xon bor ekan yolg'iz yonmadi | insertion(2), root_or_other(2) |
| 45 | `1187023182_2_37182_1` | chiqqan joyiga qaytib kirg'izvoraman xuddi | chiqqan joyiga qaytib kirg'oraman xuddi | root_or_other(1) |
| 46 | `909734087_2_30062_1` | boshqalar qani deb so'rashsa shisha topshirgani ketishdi debdi | qanaqa deb so'rashsa shisha topshiriqdan ketishdi debdi | suffix_error(2), deletion(1) |
| 47 | `1187023182_2_38313_1` | zinatulin dorga emas jinnixonaga mahkum edi | xizmat to'lin dorga emas singa | root_or_other(2), deletion(2), insertion(1) |
| 48 | `1187023182_2_35431_1` | instagram meyvezer hayotida ijtimoiy tarmoq yoki kamera qarshisidagidek emas | instagram meyvezer hayotida ijtimoiy tarmoq yoki kamera qarshisidagi emas | suffix_error(1) |
| 49 | `1187023182_2_38488_1` | hozir qaldirg'och qaldirg'och mo'ylabli yigitning yana bir marta siz ozodsiz deyishni kutdi | hozir qaldirg'och qaldirg'och mo'ylovli yigitning yana bir marta siz orzusiz deyishini kutdi | suffix_error(2), root_or_other(1) |
| 50 | `1187023182_2_35443_1` | ovqatlanadi barchaga choy chaqa beradi | ovqatlanadi barchasi choy chaqa beradi | suffix_error(1) |
| 51 | `1187023182_2_38407_1` | radioni eshitayli | radion eshitaylik | suffix_error(2) |
| 52 | `909734087_2_91141_1` | ustida iz qoldirib tokcha tomonga o'tishdi | ustida iz qoldirib tokcha tomon o'tishdi | suffix_error(1) |
| 53 | `1187023182_2_38342_1` | atrofga alangladi da qo'lini paxsa qilib jahl bilan gapira ketdi | atrofga aylanganida qo'lini paxsa qilib jahl bilan gapira ketdi | deletion(1), root_or_other(1) |
| 54 | `265773029_2_74991_2` | niyoz atrofni kuzatgan bo'lib | yoz atrofni kuzatgan bo'lib | root_or_other(1) |
| 55 | `1187023182_2_37437_1` | bilchillab eriyotgan qorlarni bosib ko'lmaklarni chetlab o'tib qorovulxonaga ham olib kirdi | pilchillab eriyotgan qorlarni bosib ko'lmaklarini chetlab o'tib qorovulxonaga ham olib kirdi | root_or_other(1), suffix_error(1) |
| 56 | `909734087_2_81819_1` | xullas mashina gadoy topmas ko'cha | xullas mashina gado topmas ko'cha | suffix_error(1) |
| 57 | `1187023182_2_35175_1` | ushbu ko'tarilish amazon'ning muvaffaqiyatli choraklik hisoboti tufayli kuzatildi | ushbu ko'tarilish amazonning muvaffaqiyatli chorali hisoboti tufayli kuzatildi | apostrophe_error(1), suffix_error(1) |
| 58 | `909734087_2_29890_1` | bolaga tegman deb baqirdi dadasi | bolaga teng deb baqirdi | root_or_other(1), deletion(1) |
| 59 | `1571110404_2_19305_1` | hammayoq dov daraxt ariq | hammayoq dovraxt ariq | deletion(1), root_or_other(1) |
| 60 | `1571110404_2_36616_1` | shilimshiq deb laqab olgan jalolni avval daraxtga bo'ynidan bog'ladilar | shilimshiq deb laqab olgan jalolini avval daraxtga bo'ynidan bog'ladilar | suffix_error(1) |
| 61 | `1187023182_2_35582_1` | bu real dagi eng sevimli lahzalarim | bu realdagi eng sevimli lahzalarim | deletion(1), root_or_other(1) |
| 62 | `1187023182_2_38480_1` | qaldirg'och mo'ylovli yigit sharifga faqiriga mensimay qaragan podsho kabi ilkis nazar tashlab qo'ydi | qaldirg'och mo'ylovli yigit sharifga faqiriga mensimay qaragan podsho kabi ekis nazar tashlab qo'ydi | root_or_other(1) |
| 63 | `1187023182_2_35375_1` | buning ustiga bizning developerlik tajribamiz ko'rsatishicha uy yo'lga juda yaqin joylashmasligi kerak | buning ustiga bizning developerlik tajribamiz ko'rsatishicha u yo'lga juda yaqin joylashmaslik kerak | root_or_other(1), suffix_error(1) |
| 64 | `1571110404_2_36699_1` | to'lg'onishga ham quvvati qolmagan jalolshilimshiqning chap ko'ksiga pichoq sanchdi | to'lg'onishga ham quvvati qolmagan jalosilimshiqning chap ko'ksiga pichoq sanchdi | root_or_other(1) |
| 65 | `1187023182_2_38136_1` | idorada charos ko'zlari o'ynab turuvchi bir qiz ham ishlardi | idorada chorasi ko'zlari o'ynab turuvchi bir qiz ham ishlaydi | root_or_other(1), suffix_error(1) |
| 66 | `909734087_2_84081_1` | ya'ni mening | ya'ni meni fikr | insertion(1), root_or_other(1) |
| 67 | `709378413_1_30286_1` | oldi peshayvon ikki xonali o'sha eski uyiga borib chiroqlarni yoqmay tanchaga cho'g tashlamay tong ottirardi | olti peshayvon ikki xonali o'sha eski uyiga borib chiroqlarini yoqmay tanchaga chog tashlamay tong ottirardi | root_or_other(1), suffix_error(1), apostrophe_error(1) |
| 68 | `351600865_2_79114_1` | keyinchalik ushbu binoda men | keyinchalik ushbu binoda ming | root_or_other(1) |
| 69 | `909734087_2_92618_1` | meni savodsizga chiqardi | o'tirsam meni savodsizga chiqardi | insertion(1) |
| 70 | `1187023182_2_38413_1` | direktor xonasidagi radio keng dahlizning o'rtasiga olib chiqib qo'yilgan | direktor xonasidagi karavot keng dahlizning o'rtasida olib chiqib qo'yilgan | root_or_other(1), suffix_error(1) |
| 71 | `1571110404_2_36809_1` | feodalizm sarqitlariga qarshi kurashayotgan el og'asi yangi rahbar kadrlar bilan band edi | feodalizm salqitlariga qarshi kurashayotgan el og'asi yangi rahbar kadrlar bilan band edi | root_or_other(1) |
| 72 | `1187023182_2_37072_1` | eski joyidami buning dedi u o'zicha | eski joyidami buning dedi o'zicha | deletion(1) |
| 73 | `909734087_2_29762_1` | alqissa yorug kunning qadriga yetmoq uchun qorong'ulikni ham ko'rish lozim emasmi | arqisda yorug kunning qadriga yetmoq uchun qorong'ulikni ham ko'rish lozimmi | root_or_other(2), deletion(1) |
| 74 | `1187023182_2_35318_1` | o'zbek va dat biroz turk tillarida so'zlasha oladi | o'zbek va tatproz turk tillarida so'zlasha oladi | deletion(1), root_or_other(1) |
| 75 | `1187023182_2_35187_1` | muhokama yakuniga ko'ra tuman hokimining o'rinbosari | muhokama yakunida ko'ra tuman hokimining o'rinbosari | suffix_error(1) |
| 76 | `818938633_2_74083_1` | bu mag'rurlikning hiyla nayrangidir | bu mag'rurlikning xiyla nayrangidir | root_or_other(1) |
| 77 | `1187023182_2_37234_1` | ba'zilar bu alamini ichiga yutib sabr qilib yashaydi | ba'zilar bu alamni ichiga yutib sabr qilib yashaydi | suffix_error(1) |
| 78 | `909734087_2_31285_1` | bir umr kibru nazokat bilan yashab bo'ladimi axir | bir umr kibri nazokat bilan yashab bo'ladimi axir | suffix_error(1) |
| 79 | `1187023182_2_38120_1` | xotinidan so'ng u ham o'rnidan turib yuvindi | xotindan so'ng u ham o'rnidan turib yuvindi | suffix_error(1) |
| 80 | `1187023182_2_35377_1` | yana bitta nozik tomoni shundaki bu avtomobil harakatining yo'nalishi | yana bir nozik tomoni shundaki bu avtomobil harakatining yo'nalishi | root_or_other(1) |
| 81 | `909734087_2_91215_1` | shodlikni bilishsa ham bilmaganga olishadi | bilishsa ham bilmaganga olishdi | deletion(1), suffix_error(1) |
| 82 | `972644779_2_78980_1` | ammo prognozlar afrikada buning uddasidan | ammo prognozlar afrikada buning uddasi | suffix_error(1) |
| 83 | `1187023182_2_35466_1` | avvalroq mayk tayson boksga qaytishi va bir nechta ko'rgazmali jangda ishtirok etishi ma'lum bo'lgandi | avvalroq maykl tayson boksga qaytishi va bir nechta ko'rgazmali jangda ishtirok etishi ma'lum bo'lgandi | suffix_error(1) |
| 84 | `1571110404_2_19285_1` | u otini to'xtatib orqasiga o'girildi va jiyron qashqa ot mingan navkarni yoniga chaqirdi | u otini to'xtatib orqasiga o'girildi va jiyron qashqir ot mingan navkarni yonga chaqirdi | suffix_error(2) |
| 85 | `709378413_1_83034_1` | ularning ikkisi ko'chaning u boshida | ularning ikkisi ko'chaning u boshida ikki | insertion(1) |
| 86 | `909734087_2_90346_1` | ammo vinzavodga osila ko'rma | ammo vindga ozila ko'rma | suffix_error(1), root_or_other(1) |
| 87 | `1187023182_2_38425_1` | stalin o'lgan bo'lsa endi urush boshlanadi dedi jalil | sistaning o'lgan bo'lsa endi urush boshlanadi dedi chalib | root_or_other(2) |
| 88 | `1187023182_2_36946_1` | senga tseytnot xavf solib turibdi | senga tsnut xavf solib turibdi | root_or_other(1) |
| 89 | `1187023182_2_38446_1` | osh posh pishirguncha eringiz uyingizga kirib boradi | osh bosh pishirguncha eringiz uyingizga kirib boradi | root_or_other(1) |
| 90 | `1187023182_2_38386_1` | asad nima uchun deb so'rab o'tirmay jalilnikiga chiqib ketayotganida orqasidan onasining so'zlarini eshitdi | asad nima uchun deb so'rab o'tirmay jalanikiga chiqib ketayotganida orqasidan onasining so'zlarini eshitdi | root_or_other(1) |
| 91 | `1187023182_2_37153_1` | pulga yollovdim | pulga yolvordim | root_or_other(1) |
| 92 | `818938633_2_74164_1` | yuz yil shon sharaf va | yuz yil shon sharafi | deletion(1), root_or_other(1) |
| 93 | `1187023182_2_37453_1` | safarga ketganlar dedi gapga aralashmay o'tirgan qorovul yigitlardan biri | safarga ketganlar dedi gapga aralashmay o'tirgan qorong'i yigitlardan biri | suffix_error(1) |
| 94 | `909734087_2_29087_1` | u odam o'n yetti yildan beri ruhiy xastaliklar shifoxonasida ro'yxatda turarkan | bu odam o'n yetti yildan beri ruhiy xastalik shifoxonasida ro'yxatga turarkan | suffix_error(2), root_or_other(1) |
| 95 | `909734087_2_28990_1` | bilagingni ko'rsat dedi zohid tirtiqqa | bilagingni ko'rsat dedi zohid tiriqqa | root_or_other(1) |
| 96 | `795967678_1_15926_1` | jonidan o'tmasa jonim demas | jonidan o'tmasa jonim temas | root_or_other(1) |
| 97 | `1187023182_2_35341_1` | xonzodabegim a'zamova suhbatlashdi | xonzoda begim azamova suhbatlashdi | insertion(1), root_or_other(1), apostrophe_error(1) |
| 98 | `265773029_2_79857_1` | germaniyaning berlin shahridagi uylaridan biri balkonidagi eng | germaniyaning berlin shahridagi uylardan biri balkonidagi eng | suffix_error(1) |
| 99 | `709378413_1_83033_1` | yangi yil kechasi uning eng sara eng | yangi yil kechasi uning eng sara | deletion(1) |
| 100 | `1187023182_2_36978_1` | uning qo'li ishda bo'lsa ham xayoli elchinda undan ko'z qirini uzmas edi | uning qo'li ishga bo'lsa ham xayoli elchinda undan ko'z qirini uzmay edi | suffix_error(2) |
