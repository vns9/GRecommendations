import matplotlib.pyplot as plt 
import numpy as np

list1 = [40,41,42,43,44,45,46,47,48,49]
List2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
list2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
list21  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
list50 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
list100 = range(100)

## Without Dropout
#test1
b1 = [0.2462264857527667, 0.23713576226037536, 0.22878189504585228, 0.22048013509771813, 0.21264177899624082, 0.20509861503342883, 0.19946613492175752, 0.1944689874330653, 0.19105777025657256, 0.18820477695753376, 0.18670690923294456, 0.18599612910590488, 0.18558194258951127, 0.1847774704036594, 0.1843691806405947, 0.18389033924077836, 0.18361551179566382, 0.18353553780276746, 0.18327279641893487, 0.18305660763419263, 0.1829750455616853, 0.18222124780689422, 0.18212460928035543, 0.18213849076226618, 0.1820605024633301, 0.18172919813485064, 0.18188345301359632, 0.18157527872642756, 0.18155144737602774, 0.1812925225074772]
bah1 = [0.2927458716231415, 0.27594691900574364, 0.261614350230026, 0.2501475971142288, 0.23920634694781898, 0.23025671393055608, 0.22097277201866855, 0.21389231544628637, 0.2079046588214017, 0.20327360703556133, 0.19914424995493768, 0.19592157948235167, 0.19376873947286608, 0.19172456104888005, 0.190674796134277, 0.18997489658423292, 0.18950897813515297, 0.1887970586724519, 0.1882130819087418, 0.188036773995408, 0.18759560257015234, 0.18760117571465978, 0.1871943631449813, 0.18685456642730008, 0.1866017881137844, 0.1867389944126579, 0.18660666574595144, 0.18591292819482538, 0.18594160395760842, 0.18631538587988]
#test2
b2 = [0.3071540450650362, 0.2953097344810062, 0.2809379831722005, 0.26521942283135264, 0.249020935771883, 0.23429343260483018, 0.22152535919952282, 0.21158117560844408, 0.205504268697449, 0.20145849256211898, 0.19834008408104153, 0.19648040738405284, 0.1948194785311701, 0.19371496855349088, 0.19252270944917863, 0.1916683194275414, 0.19095258704398296, 0.19034713765273487, 0.18986209991850309, 0.18898749609378537, 0.1893597925744242, 0.18847512582617945, 0.18850208611270525, 0.1877703163610896, 0.18747616836581493, 0.18727280416523578, 0.18719578938315556, 0.1869110361615065, 0.18702129007885457, 0.18640034083908916]
bah2 = [0.3607331690123567, 0.334030338165903, 0.31028344503857364, 0.288378883073167, 0.2677687811112425, 0.25068001945742496, 0.2362206011930596, 0.22408211216336327, 0.21639406276945886, 0.21048520726597814, 0.20590047152532717, 0.20328408975613213, 0.20219295321916017, 0.19982457649325447, 0.19891513686046117, 0.1978570897721727, 0.19663976872812225, 0.1957882769995636, 0.19499937524371616, 0.19433808867798558, 0.19391218769400542, 0.19358782222015183, 0.1934590905578676, 0.19247622472612408, 0.1921022208897264, 0.19180044283831715, 0.1914876342493968, 0.19089275736744235, 0.1909194678663088, 0.19057658480791786]
#test3
b3 = [0.1938230120616007, 0.18891465665354445, 0.18621907014935224, 0.1850627263582125, 0.18415570561429265, 0.18368967986183418, 0.1829240925551562, 0.18247957041017557, 0.18232235755961673, 0.18181021362035357, 0.18157716636668475, 0.1815222439907675, 0.18109854740818715, 0.18094041183117543, 0.18052308604159786, 0.18036296170341126, 0.18035334889748023, 0.18005720892704558, 0.1802694069751929, 0.17993721249381456, 0.17990051923123276, 0.17968864562732004, 0.17958489994852506, 0.17983046190484286, 0.17981575911411105, 0.17980456149852134, 0.1794574327506567, 0.17928310584845064, 0.17973063085324265, 0.1792712964109306]
bah3 = [0.24588879105612013, 0.23326269380368478, 0.22236876642077483, 0.2140664501728685, 0.20748403873705268, 0.20261377503987177, 0.1990230650785769, 0.19652026822920027, 0.19448396577335306, 0.19328218687710416, 0.19241339922378883, 0.19111986040344411, 0.19019597735074315, 0.19020353944496654, 0.1892786994433116, 0.18894529978152014, 0.18821470832004342, 0.1877190565565904, 0.1877500474535163, 0.18725082217260136, 0.18700792111206643, 0.18688048609841884, 0.18617770904243194, 0.18594838656493753, 0.18588187908500214, 0.18540054063343167, 0.18535428019333897, 0.18526437903899132, 0.1852469424422102, 0.18459949039002718]
#test4
b4 = [0.3673887888084082, 0.3557945092338911, 0.34453659333266456, 0.33489514905413914, 0.3255132854014509, 0.3172387235172157, 0.3080760849886362, 0.29886017710482876, 0.28933639181967497, 0.2797078756755561, 0.27000549618698927, 0.2601219724501654, 0.25013377424414296, 0.24171857783059558, 0.23354684601673545, 0.2256225890105217, 0.2189638294787041, 0.2140099589230138, 0.20972723201046023, 0.20612303707295945, 0.203504536173896, 0.20222943739748256, 0.20057000254411236, 0.1998303518235685, 0.19880347064361556, 0.19800882066299355, 0.197020122820408, 0.19675517246470495, 0.19641868999096906, 0.195888468177988]
bah4 = [0.23511689528992666, 0.22132671614056737, 0.21066061022046365, 0.20337900851595003, 0.19839531387685364, 0.19465549804507332, 0.1926380370207416, 0.19064387243091038, 0.18983553626762634, 0.18830218943578192, 0.18752181582502536, 0.18686199809528004, 0.18648867082264448, 0.1855969173807991, 0.18504080381989046, 0.1850052105689078, 0.18445061272198598, 0.1844249143822986, 0.18418050520894505, 0.18376420824574535, 0.18348374961459768, 0.1837517431076966, 0.1831637785953944, 0.18295448048732102, 0.18318985862161202, 0.18263085808472904, 0.18238774433392, 0.18253844380098028, 0.18224223779253648, 0.18227336926539034]

#with dropout
b5 = [0.22028555466938965, 0.20849255433872366, 0.20049219192498519, 0.19429626429496175, 0.19099392882488544, 0.18858177429174675, 0.187372913140515, 0.18658050702969564, 0.18563522794526563, 0.185256464908508, 0.18440406654047176, 0.18382025831592877, 0.18338422432445778, 0.18324918911177335, 0.1828848795366264, 0.18234788789672146, 0.1818371015561727, 0.18178928133907046, 0.18178857079622276, 0.18147605863147884, 0.18174866935834386, 0.18110945176975787, 0.18112018002132307, 0.1808899959243242, 0.1810281679790971, 0.18098042524705973, 0.18132635372568146, 0.18061959923442944, 0.18072967729212017, 0.18028171191148495, 0.1801226301027014, 0.18043043764141695, 0.18023340404840058, 0.18032366619918994, 0.1802294176175485]
bah5 = [0.22718592349669217, 0.21656294847529656, 0.20868792295546731, 0.20339011513221494, 0.19973308957818148, 0.1965679383132533, 0.19497765721201302, 0.1931718574211641, 0.19233197409713743, 0.19114816753759278, 0.19004563397702398, 0.18935372684515944, 0.18889482639504684, 0.18821686482443697, 0.1878913185798584, 0.18746542391039617, 0.18694301909164404, 0.1866225143291657, 0.18654459690276773, 0.1860641238751516, 0.1860228860935545, 0.1858436829506666, 0.18539483820284405, 0.18531847940677548, 0.1848182461592013, 0.18487195931996936, 0.18456750228432003, 0.18439880443124318, 0.18422056357855135, 0.18386902483105777, 0.1836377270514231, 0.18364656356355025, 0.18350738000894234, 0.18369647835838246, 0.18330608169731927]

b6 = [0.3238072249458999, 0.3171432932324867, 0.31229539116711197, 0.3081931763931625, 0.30438658976844596, 0.3009775491766389, 0.2972729862799463, 0.29291245348521366, 0.2881382071927517, 0.2814952925635678, 0.274066896886867, 0.2650968765438478, 0.2558049878094553, 0.24673054561015875, 0.2381470938536828, 0.22963346960009481, 0.223126870177181, 0.21732548254047573, 0.21284259819392343, 0.20860113077237186, 0.20554044538540317, 0.20314678464859354, 0.20118586046607287, 0.20007806964574426, 0.19805534272938366, 0.1978560076008142, 0.1964642327150904, 0.19598642400101612, 0.19466554486249318, 0.19441583388997874, 0.19313771121855774, 0.19339106685941157, 0.19219732249216234, 0.19197195690428673, 0.19134946864283198]
bah6 = [0.23683167016281614, 0.22959063678121966, 0.22140782417902755, 0.21378029581273944, 0.20723664775100845, 0.20176487348011155, 0.19725048566080955, 0.19378048209670307, 0.19152572135028062, 0.18999925737189105, 0.1884042850927244, 0.18731315753934633, 0.1867835208682558, 0.18605598752220576, 0.18539443114214685, 0.18461754495158758, 0.18425288363464887, 0.18373312143407952, 0.1837225122408, 0.18327329036901857, 0.18285369100436036, 0.18328760070587913, 0.18278253470479444, 0.18286100330348928, 0.18229838442897034, 0.18234425573320037, 0.18212510295617687, 0.1818998483474201, 0.18182203694710586, 0.1816841182565446, 0.1814445743518113, 0.18148928034592607, 0.18164888833370896, 0.1810861563384438, 0.18157310450384254]




b100 = [0.2340442598790954, 0.22631479073346863, 0.21831456680348305, 0.21065571539812897, 0.2041104030202619, 0.19819578724254128, 0.1940096322974251, 0.1910639564691745, 0.18881510391304984, 0.18786228892521456, 0.18648511941854767, 0.186265271156339, 0.18561084412629608, 0.1853115773876883, 0.18468599973904173, 0.18442622549560475, 0.18436835496032872, 0.18398814014777115, 0.18392110831999867, 0.1835731553539941, 0.1832982925768539, 0.18328326008444676, 0.18278367658198894, 0.18294329332654172, 0.18267878301602944, 0.18280955984661895, 0.18253074349079407, 0.1826380068660722, 0.1823918145662243, 0.1821137804831985, 0.1821882204768463, 0.18187158750582888, 0.18196614231294783, 0.1815754866756586, 0.18140178045469785, 0.1814049027536078, 0.1812747206797229, 0.18133846421056155, 0.18101430908028993, 0.18114246967647563, 0.18073778985433123, 0.18105966758910938, 0.1806928084787234, 0.18045925605202545, 0.1805531851935176, 0.18037637423133818, 0.18082360945555778, 0.18066272207821557, 0.18025124166017162, 0.1804128932144555, 0.18018390408366888, 0.1801942000009924, 0.18021726146722275, 0.1801418092137186, 0.18017935363059495, 0.18004741334998894, 0.17975011919265577, 0.1799989088745019, 0.17991698432640948, 0.1799447462369911, 0.1796838815137519, 0.17967716187441157, 0.17962755801382918, 0.1799196857294788, 0.17912122556945326, 0.17934062836684936, 0.17951470752764043, 0.17937307287759024, 0.17948532602776052, 0.1790553504583988, 0.1792548809699022, 0.17945696438079276, 0.1793583221980098, 0.1793716440194619, 0.17898573609790908, 0.17908005358489212, 0.1789542273583688, 0.1790320576402075, 0.17914213680306554, 0.179291174449633, 0.17944840514187252, 0.17900939255935547, 0.17881071060675718, 0.17917843634505695, 0.17901249228132385, 0.17933917540775837, 0.17896674438567706, 0.1787759077231492, 0.17905453114025188, 0.17888487673139236, 0.17894071758955327, 0.1788558827631854, 0.17846152402601917, 0.17914049154108805, 0.1788117581514931, 0.17865142299545864, 0.1787817583871245, 0.17873635305033414, 0.17866547277652833, 0.17856834175369218]
bah100 = [0.3037877004930786, 0.2787214744582916, 0.2573919909992395, 0.24261042934248048, 0.23082221002551587, 0.22176599354901538, 0.21542675602861788, 0.2104726951428513, 0.2067305808449758, 0.20382964578243165, 0.20096666358375373, 0.19873198483405505, 0.19656210697335258, 0.19541246332429246, 0.19395234162482916, 0.1926507391912446, 0.19164011966221361, 0.19090736047539636, 0.1904224224188325, 0.19006678861954676, 0.1892442654132454, 0.18911735901282387, 0.18885341106262327, 0.1882138689392376, 0.1880126977004764, 0.1875728842350876, 0.1875490313252016, 0.187288853067107, 0.1871082085664117, 0.18660825570255316, 0.18653977254331702, 0.18636358015712712, 0.1862902094748664, 0.1857923725303084, 0.18568269916325097, 0.1854982363602971, 0.18531357422618192, 0.18553158231784025, 0.18539882664561727, 0.18532279081093553, 0.18471887830817887, 0.18482294275130687, 0.18466481892877304, 0.184410848864065, 0.18446962510568768, 0.18423854614447407, 0.18390267136443292, 0.18372558801252092, 0.18403264437687847, 0.1839594964188273, 0.18392237595507113, 0.1832955333479323, 0.1833775565041564, 0.1834091518025236, 0.18349458932774343, 0.18331245229258866, 0.18268713430256475, 0.1830713612813859, 0.1826479115198367, 0.18294186616221478, 0.1830146628379176, 0.18280350679217006, 0.1825532736754587, 0.1825520889655651, 0.18240332788474928, 0.18226391301739955, 0.18246025236419922, 0.18215291965351307, 0.18215327109923923, 0.18198907320584987, 0.1820137564148727, 0.18206387172993257, 0.1819235772434475, 0.18172093085706212, 0.18180679987481332, 0.181740889507454, 0.18173544788379248, 0.18151135435114782, 0.18161035348392066, 0.1815006371110169, 0.18163555262079628, 0.18148611171173962, 0.1814012901801348, 0.1817008908295057, 0.1815176274831853, 0.18150893465090645, 0.18131183433973894, 0.18082114396544213, 0.18095861927251977, 0.18114591442388894, 0.1809133065977086, 0.18128448749988693, 0.180821706548985, 0.18080755823323771, 0.1809996849256381, 0.18081847239244153, 0.18084417970318767, 0.18103481271380703, 0.18061738514650438, 0.18074563078886588]

#camra dataset
bil = [0.23670902503127464, 0.2304961756136831, 0.22578311631035808, 0.22246117198645424, 0.22022546385493122, 0.2185551802146123, 0.21737160307105236, 0.21650764745343615, 0.21588433712307603, 0.21547241940152947, 0.2152198132832694, 0.21481925741499736, 0.2147053939318265, 0.21456323825742193, 0.21430480254508855, 0.21422476008966096, 0.21427408937189932, 0.2142505656133225, 0.2144208702372071, 0.21426438822866198, 0.21429225487771428, 0.21426848111391508, 0.21394994135384857, 0.21396241778150862, 0.21388688080646395, 0.21384710820923866, 0.21391484290175636, 0.2138188570041645, 0.21364004737819234, 0.2136954474983187, 0.21368923753554925, 0.2136923345084562, 0.21366853301484362, 0.21361960859976767, 0.21366624222000466, 0.21368368666806387, 0.2135722366071395, 0.21338217090509393, 0.21341645230681397, 0.21364252614988608, 0.2136120774300121, 0.2134395166132614, 0.2136469596688766, 0.21358667464312972, 0.21361505440794828, 0.21346513440637196, 0.21350589779854268, 0.2132376355092395, 0.21342589858157698, 0.21338569407092298]
bah = [0.279606026122061, 0.26035236771864145, 0.24676921808474986, 0.23730474308225516, 0.23100499843850075, 0.2268204936518317, 0.22390088902278651, 0.2222150440986773, 0.22073049551046656, 0.21966208002196058, 0.21872477693797282, 0.21820751681630307, 0.21756595354536823, 0.21728838234641604, 0.21684218045808715, 0.21668212534129672, 0.21667198579338331, 0.21665666665976951, 0.21653469703814304, 0.21648167554273612, 0.21627363142464004, 0.2160087229841545, 0.21598233797340355, 0.21574026074367642, 0.21558614347953522, 0.21548189108676552, 0.21538305945581174, 0.21534908400382757, 0.21510157344613326, 0.21490968164653176, 0.21492819187140644, 0.21461272527039305, 0.2147155349286185, 0.2145304579637859, 0.21450854724195076, 0.21460986538021998, 0.21443978798053961, 0.21438556910110257, 0.2143816132936039, 0.2144012750368646, 0.21421082959474325, 0.21443543216888536, 0.21417196868301372, 0.21407276823234972, 0.214222814221075, 0.21416239653463773, 0.21423970851409385, 0.21401226048090805, 0.2140451074809301, 0.21412273135504256]

bil = [0.24563276017675675, 0.23776119532147616, 0.23164947953882709, 0.22743193818286003, 0.22452041300014405, 0.22245997676480947, 0.22135814678634247, 0.22058390404598424, 0.21979157628055337, 0.21926031334360482, 0.21891595085135815, 0.21850942035810691, 0.21835849962533338, 0.21801935405741166, 0.21798776593065103, 0.2176273762021575, 0.21758494884947197, 0.21760026375155972, 0.21755169465196247, 0.21754677749862159, 0.21764043828421475, 0.21740438109355442, 0.21733082557410072, 0.21719047538206068, 0.21719884784368262, 0.21691877944941662, 0.21677048251041697, 0.21693225714847883, 0.21667044153116213, 0.2166840609522051, 0.21661338578780182, 0.21658074300939947, 0.2163814683820312, 0.21642650379222406, 0.21628833867893252, 0.21625548935773622, 0.2162592897709252, 0.21616484675025402, 0.2161743032011749, 0.21612456276202974, 0.2161564799814913, 0.2160901922471914, 0.21604159654785116, 0.21603083294049624, 0.21589195795221972, 0.21590852769466817, 0.2159245047302142, 0.21587736517799289, 0.21595435415559278, 0.21580423414116973]
bah = [0.32441360853020246, 0.29773048353771625, 0.27569260968003667, 0.25920227727867573, 0.24751435695921148, 0.23991760597746492, 0.23463211279691346, 0.23062613893456285, 0.22775827095821613, 0.22522769424184644, 0.22349327470322813, 0.22192752902579133, 0.22074512527477688, 0.21992361618686695, 0.21905166830100278, 0.21886851868676035, 0.21878321504533246, 0.21864299301762213, 0.21853016986862597, 0.2184363974947813, 0.2180797700820542, 0.2175936987781849, 0.2172215120317806, 0.21713008274975729, 0.2166870363555364, 0.21646557684366008, 0.21620856600742275, 0.21601658761808523, 0.2159955243577501, 0.2158267834295757, 0.2156133383874087, 0.2154887000703237, 0.21535446907753067, 0.21537229087286414, 0.21526175835680875, 0.21510259801516377, 0.21503826701583456, 0.21495865650079027, 0.21504085806044937, 0.2149544997232977, 0.21487749822210303, 0.21479326986904884, 0.2146885024429121, 0.2146800103015212, 0.21461973710714768, 0.21456976738236383, 0.214538278542511, 0.21462884688748096, 0.21437053599236264, 0.21440761413107598]

#smaller dataset
bil1 = [0.510358930184565, 0.5112469320933223, 0.5110512841442469, 0.5093621413152515, 0.5079206722600896, 0.5069603697865885, 0.5072165884607864, 0.5040723936981686, 0.5039243590333505, 0.5003006814878003, 0.501737006343518, 0.5038298893286276, 0.4999614674503872, 0.5028327358710464, 0.5003992884870822, 0.5000930896691314, 0.4951821182943661, 0.4955126890065029, 0.49585005848485086, 0.49435317329955825, 0.4922297284157424, 0.4927941048233613, 0.4944347234625807, 0.4919936546584142, 0.49053890278035905, 0.4857817598500143, 0.4845033288382119, 0.486237752715731, 0.48553724449351504, 0.4834449944013921, 0.488928392181537, 0.4862670412650032, 0.48486248765861645, 0.48569685131713597, 0.4795714354162804, 0.4889075907220842, 0.485487991340363, 0.4747724820151336, 0.4784468956989347, 0.4781084463061306, 0.48075693537727865, 0.4741664817646073, 0.4759143164344386, 0.4744622062881192, 0.4742934590066666, 0.473953068627262, 0.4711232139371618, 0.4675415629721777, 0.4733118372783514, 0.46705983147553715]
bah1 = [0.7049229567152441, 0.6953521609076763, 0.6858576365106744, 0.6941871138076947, 0.7058625260230302, 0.692264108505522, 0.6808684085208097, 0.6744415603493971, 0.6773408372155968, 0.6709232770280849, 0.678562012572393, 0.6678819202267375, 0.6668937295883576, 0.6641380833665371, 0.6654539956102274, 0.6541244369606974, 0.6360667055462564, 0.6385967290891582, 0.6460001486806146, 0.6334421249164638, 0.6354924104320034, 0.6289622402032146, 0.6225334956999425, 0.6220513336370229, 0.6115436179271379, 0.6096616431478831, 0.6002161521782828, 0.6101396027593771, 0.6175989106126514, 0.6125547840853889, 0.5964248775867316, 0.6046905012742799, 0.5827289744251004, 0.5850732748632048, 0.5881480791283388, 0.5737687630746603, 0.5763069193540976, 0.5787891853769931, 0.5709478570355624, 0.5667356340474871, 0.5615949206522662, 0.5576791162419322, 0.5585479694779039, 0.5509757912425756, 0.5427506151372746, 0.5500566573787077, 0.5353175634489472, 0.5480017542383229, 0.5511959960056778, 0.5507849359007029]

bil2 = [0.5387153457345686, 0.5380113054621176, 0.5365248367671264, 0.5365374084979695, 0.5363879287018845, 0.5346283586056074, 0.5340685833870735, 0.5347171106737342, 0.5328337009573925, 0.5330976686577453, 0.5325801305288326, 0.529615875030105, 0.5310763829197108, 0.5312784237409589, 0.5303035654451195, 0.5300970902552524, 0.5274405435130958, 0.5283589796033636, 0.5262933196749868, 0.5273694357541608, 0.5268864656135562, 0.526249777068286, 0.5243333865591935, 0.5251206615269337, 0.5230220097376665, 0.522786232029596, 0.522834468904191, 0.5232801703998178, 0.5227477584032996, 0.5221258756005417, 0.5219001751629242, 0.5202861904124968, 0.5212601608675239, 0.5207346896857835, 0.519878721520815, 0.5203130207739064, 0.5191653149004316, 0.5182914349274393, 0.5177563447589941, 0.518181495753231, 0.516985992002892, 0.5163843430788568, 0.5172258519357409, 0.518419876924505, 0.5171477125628827, 0.5151206532818905, 0.5154552772847385, 0.5149441867924529, 0.5152443837454542, 0.5141948296002999]
bah2 = [0.5574687911115628, 0.5567431596827258, 0.5647510552585817, 0.5492473565015412, 0.5540662488584926, 0.5405839349722721, 0.5342282687512265, 0.5465776704529542, 0.539567220046596, 0.5351977772913885, 0.528479422299269, 0.5332812837152817, 0.5238676998133514, 0.51759498235456, 0.5169960163434882, 0.5097174019025003, 0.518555812088736, 0.5151965578146185, 0.5067895400630498, 0.5110746999529368, 0.4928146645450991, 0.5036553933781382, 0.4895920165565295, 0.5005083404356816, 0.4889546735689215, 0.4893754981926054, 0.48138468516182964, 0.4895811989990302, 0.4798792525186819, 0.47359394332811594, 0.4749739377210978, 0.4623168585405579, 0.45205300078850685, 0.4677706892968433, 0.46561492706939633, 0.4638649139966745, 0.45164231767812246, 0.46647093300688053, 0.4404228750437549, 0.4485701875157118, 0.443824373690747, 0.4478205052057469, 0.43545869947964644, 0.4323279602737285, 0.44242577093017116, 0.42541387525765045, 0.43106693095047355, 0.4367045748602765, 0.4239678042294716, 0.42775983831814873]

# -1 unknown
bil = [0.631059039589565, 0.6129589605030081, 0.5980355074227495, 0.5856547515106509, 0.5754280791996372, 0.5660366990436629, 0.5563815182622038, 0.5462894155733061, 0.5356364279400635, 0.5240373752450015, 0.511302693560195, 0.49850788244777333, 0.48501057298341954, 0.47109961076889534, 0.45711567699923533, 0.4423011119294184, 0.42780658356721024, 0.4125337455599832, 0.39768570934940917, 0.3810819572924866, 0.36606657151853295, 0.3500698524234152, 0.33391078868084606, 0.32033951507495834, 0.3037910974188396, 0.28845311632904774, 0.27329830931708854, 0.25892180965329187, 0.24470305870722261, 0.23150341647040915, 0.21848862887607884, 0.20612868201111698, 0.19530620141496874, 0.18394469668270108, 0.17357500132515669, 0.16462827997733429, 0.1547239894009279, 0.14813057423773604, 0.13869214545774064, 0.13216340046388356, 0.12552476600503076, 0.11927541956689458, 0.11297388397259725, 0.10857559332694924, 0.10351421709140105, 0.09974051172417896, 0.09274100296260904, 0.08980425084318036, 0.08502004959979859, 0.0836347180533649]
bah = [0.4412904088089794, 0.41208611294148373, 0.38428998465106523, 0.3586961362755899, 0.3342560453179828, 0.31208639938067345, 0.2908991435692292, 0.273042650656478, 0.25564348716148183, 0.2419751533948488, 0.22777216478288567, 0.21563283943951092, 0.20555175836392028, 0.19629936315378438, 0.186232254266859, 0.17711157660730564, 0.17092918859642775, 0.16289383520989992, 0.15650831505684176, 0.1518004620341303, 0.1456475982695258, 0.13951085498728633, 0.13360655183315662, 0.13170542059767834, 0.1260760824484437, 0.12282201420131082, 0.11928924717221082, 0.1166767176326241, 0.11216394204814926, 0.10954336364832397, 0.10658817971460524, 0.10541180463134474, 0.10264093679446804, 0.10265066740859723, 0.09909949980649349, 0.09673413629249701, 0.09511398135515167, 0.09593317269878969, 0.09208404671881054, 0.09280972124996079, 0.09147162459308533, 0.08919318159086198, 0.08900365427386447, 0.08713162609323434, 0.08639237120794496, 0.08477544785439942, 0.08351341403374203, 0.08234676805593377, 0.08396542481338333, 0.08239624559503643]

# -1 unknown 100
#bil = [0.46426941894781426, 0.44403683800706123, 0.4241996270469674, 0.4046268192476946, 0.38561035703704993, 0.368531200194771, 0.35226206375457386, 0.3363620144442261, 0.32171163678331954, 0.3071550447917593, 0.29412379301485103, 0.2812182961686039, 0.2707627991688968, 0.26013619361657264, 0.24922712641504965, 0.24142742686471586, 0.2339673536902466, 0.22624370871087623, 0.22090429375585144, 0.2153340920507077, 0.20945275593293924, 0.20608166618004356, 0.2042184435571559, 0.20128027995021458, 0.1968015956521399, 0.19573243589283526, 0.19349537378442228, 0.19470186429538405, 0.1920431082800118, 0.19007687802884748, 0.18972720074810986, 0.18986142323910207, 0.18797596277405126, 0.1894897268319715, 0.18496687657338898, 0.18988770551639556, 0.18566060361272976, 0.18385489939623534, 0.1842145745963972, 0.1842541682217603, 0.18328919904801608, 0.1839517517441176, 0.1836288790176339, 0.1825235479326957, 0.18244973965725275, 0.18569240567099637, 0.1831435737664782, 0.18101482233601243, 0.18085257753503167, 0.18064788505307708, 0.17767697668840707, 0.1790047758343854, 0.1809556549613862, 0.17859826881714266, 0.17975990541047332, 0.17959337050302546, 0.17975038678276128, 0.17704103739243132, 0.1779390387583612, 0.17722766779774693, 0.1780440096067962, 0.17682124874805716, 0.17505504457348264, 0.17590907166349107, 0.17630231938640134, 0.1763829885659244, 0.1764151029002341, 0.17557245277050793, 0.17645119310559074, 0.1766736998985523, 0.17461534361945474, 0.1753113616358794, 0.17525498022136315, 0.17343448183674026, 0.17260192240365138, 0.1702726422804367, 0.17451423595124166, 0.17185853951823532, 0.17100072193389226, 0.17139533996313816, 0.1705341230978778, 0.1703621246819058, 0.16888429524357604, 0.17183054058056965, 0.16962935282287517, 0.17045360561000353, 0.17010149041955855, 0.16904318124655848, 0.16907704026652695, 0.16935969728328673, 0.1705640672251, 0.16633576011078982, 0.16775373908205474, 0.1671228904816069, 0.16689399272094932, 0.16678430531939994, 0.16746265591101409, 0.16545606322115783, 0.1667346323964792, 0.16513552633988987]
#bah = [0.5335134476054996, 0.51075494514664, 0.48613265991901444, 0.46139585217888135, 0.4349504471304225, 0.40881922127338327, 0.3829114391674032, 0.3568805089219663, 0.3331181601893592, 0.31101287359878327, 0.28924098970267925, 0.2710971147615252, 0.25145852811440333, 0.2357734979709283, 0.219966905126018, 0.2049635281120806, 0.1931662577165981, 0.18133035670963632, 0.17051109448709864, 0.1603352737282579, 0.1507269966731152, 0.14361751161761938, 0.1360870751130337, 0.1276312179118723, 0.12119702057680752, 0.11551990267938252, 0.10924897220934866, 0.10328917949189843, 0.10035572807867049, 0.09408962748916626, 0.08924540606771968, 0.08524851679828407, 0.08045722795515362, 0.07699169225804307, 0.07224067735270538, 0.07032033135579709, 0.06644931756838868, 0.0642758260204829, 0.062362405334084846, 0.05912783164565492, 0.05603566930141905, 0.0543573604953724, 0.05109540854072098, 0.04802257770260795, 0.04674943605285371, 0.04518446476095595, 0.0433108300510974, 0.04114940662235521, 0.0396047859564127, 0.038231506871341706, 0.03735840738249558, 0.035777025704296125, 0.03314227349165531, 0.03292274308280079, 0.03147140563231539, 0.029566760811115585, 0.030190574030529275, 0.028744791590649067, 0.02792813803071114, 0.025670145991805112, 0.024980963074238877, 0.024681138351064745, 0.02159052754569665, 0.022844543416185657, 0.021147534905683503, 0.021785115624930993, 0.021593195662599256, 0.020623750693285665, 0.018928599500458332, 0.019658751801613205, 0.018296189913802652, 0.01841786344034704, 0.018198571457021008, 0.015869474969232265, 0.015408183992916233, 0.017083723049536557, 0.015707066350171116, 0.016257006349338854, 0.014881825544513594, 0.014702921967512845, 0.014797491409717008, 0.013386819989427054, 0.012456841688599878, 0.013731501140149625, 0.01312572386567415, 0.013325800445838816, 0.012544583075674116, 0.01206100601674222, 0.012459834111906714, 0.01131334258321762, 0.011184368584683423, 0.010284207663257078, 0.012812204746349441, 0.010951318293599719, 0.010428462222310886, 0.008594747635702574, 0.01068486798099522, 0.010273003799587207, 0.011101566202414008, 0.010199755381803973]

list60 = range(60)
# 60 epoch bil+linear
bil = [0.28502657355840405, 0.26411516158478804, 0.24581583859164718, 0.2311375622940178, 0.22196997814894917, 0.21482741715910764, 0.21142249098892243, 0.2080125345134735, 0.20554523558550356, 0.20356987845034175, 0.20153778951190895, 0.2009202361200125, 0.1995208940777825, 0.19815865036640073, 0.1967091815865102, 0.1956695580309585, 0.19431815283999815, 0.19353328743656717, 0.19277222534625935, 0.1916455081233237, 0.19072300435886064, 0.19017248979293017, 0.18966253419088605, 0.18849766742019594, 0.18829911456484016, 0.18805979334534115, 0.1872269919069579, 0.186756324842227, 0.18663583913460138, 0.18626865926476494, 0.18610333418200017, 0.18612183310587924, 0.18549004745219455, 0.1854583098174383, 0.1861458153303888, 0.18553762887195654, 0.18526550125928204, 0.18513211660212453, 0.18526267261351928, 0.18509430591194645, 0.18479011861016995, 0.18489027367418992, 0.18471819127790715, 0.18460413905591821, 0.1843028768579989, 0.18505589417975027, 0.1843617394690525, 0.1845940299822037, 0.18418421821783515, 0.18386845919087863, 0.1842076566941002, 0.1838911209556111, 0.18387376606278372, 0.18392649361443764, 0.18368748214483163, 0.18390742722479644, 0.18420505674016716, 0.18320174470425854, 0.18362732931409836, 0.18350746079763952]
bah = [0.267541606580645, 0.2527091906779716, 0.23874083667508691, 0.2278881218889834, 0.21895985427453396, 0.2125299765971915, 0.20799391825135805, 0.20494915386315748, 0.20155087027618213, 0.19931471803335168, 0.19744524218161813, 0.1957566106073606, 0.1951290809191399, 0.19345279993919418, 0.1923428942673007, 0.19178623320597366, 0.19146833349904477, 0.19027388138622872, 0.19000787534826163, 0.18904042302721863, 0.18905629674754662, 0.18876787504816978, 0.1880651679256826, 0.18748483701402308, 0.18673579240507246, 0.1863341181539976, 0.18583370787923098, 0.18565221840126106, 0.18500571616302963, 0.18460516508330577, 0.18442383121448502, 0.18414715649191588, 0.18392825949254232, 0.1834876476624545, 0.1832757881021281, 0.18286952715683374, 0.1827257598907145, 0.1827744402230435, 0.1823842050065465, 0.18250782455031353, 0.18220452529344863, 0.18201509863726803, 0.18202855973588372, 0.1817826020333812, 0.18179103821124631, 0.1815908094308126, 0.18142233556310308, 0.18145775882227683, 0.18186441366776826, 0.18129054243623613, 0.18145251935605292, 0.18129262784728786, 0.18119318491235675, 0.1814147792138758, 0.18124768417111203, 0.18130730319925528, 0.18103838837912448, 0.18090064447572313, 0.18096006754683763, 0.18124548583609998]


#100 loss
bil = [31.400289938978926, 30.674480790363273, 29.904962178191262, 29.157637176078975, 28.096915232558406, 26.969222067354448, 25.684264200508693, 24.38511756511795, 23.10789781206938, 21.93253881930297, 20.780840552479212, 19.88855592600073, 19.057959921992296, 18.391726128360315, 17.900931032362806, 17.434401469728655, 17.09794085613441, 16.80931261440825, 16.601330278384662, 16.41685172362114, 16.248247703481983, 16.102483305703466, 16.025795922792692, 15.906395179054707, 15.839695886741978, 15.726208195772035, 15.655044396866456, 15.574592024634692, 15.557021507503812, 15.516696027430052, 15.43319641690899, 15.393976693330023, 15.35228604431458, 15.291223848023291, 15.303287460878497, 15.239948181152144, 15.22278759449902, 15.20764698508419, 15.198451748805244, 15.160354906155074, 15.193206462706076, 15.15764325083037, 15.099522437193988, 15.06577877185903, 15.062018182764506, 15.04928442419491, 15.068819114552086, 15.01204920829876, 15.001700141346237, 14.98953542686592, 14.999478704172232, 14.988591609574142, 14.95968065733941, 14.94804290598607, 14.909316196530918, 14.957004321187469, 14.954471591352924, 14.906490818003272, 14.899543090915774, 14.884594056012807]

#no att

bil = [35.207168156485636, 34.45614924771404, 33.70209131342415, 32.957835062281156, 31.97006827508247, 30.94508709630908, 29.819962404013605, 28.655845249127257, 27.438856595514075, 26.26621662571684, 25.100014478126596, 24.200378247292996, 23.353861656689865, 22.67615582902864, 22.178283731336837, 21.672873949385473, 21.31669270825296, 21.000462472628275, 20.778459734075053, 20.59074305746672, 20.403884423348213, 20.25685715811003, 20.175792989642403, 20.060521260346533, 19.985265578486274, 19.84742954516004, 19.791987880569078, 19.69331754529022, 19.676805063825636, 19.62638045019967, 19.536262768494122, 19.486433290763774, 19.442089902279417, 19.353078998729334, 19.388720898212366, 19.30403262378741, 19.29089390018924, 19.251553075420674, 19.23591374425285, 19.190444473934495, 19.2050417926188, 19.176346159931825, 19.091476563801116, 19.04098012615154, 19.03610077688382, 19.01110951259489, 19.00512016896743, 18.950418431256132, 18.932957536011607, 18.9147971681151, 18.90556470401469, 18.883498675671685, 18.85112405992756, 18.830581288120584, 18.788988053704937, 18.819467717995952, 18.824174462674286, 18.76741894320538, 18.755989297800202, 18.74361378027443]
bah = [65.40004943490884, 64.56038331177348, 63.595454033525144, 62.525572288438816, 61.52669207163849, 60.46422655269875, 59.38050043772325, 58.291002406212414, 57.22104795020166, 55.960799695169925, 54.76851595016954, 53.57489319854428, 52.557386223091406, 51.176993738673076, 50.02598195411158, 49.14049906676432, 47.94162565317464, 46.82352152495564, 45.864436636713954, 44.88482162415389, 43.87859417277194, 43.091413999413476, 42.07411627994206, 41.343033098264506, 40.511367499169886, 39.85138441037803, 39.16775066723799, 38.477251406042726, 37.84814960749192, 37.31070104905412, 36.740138214814145, 36.11889413698733, 35.741682580076564, 35.28174399468587, 34.79446659369201, 34.4509677464412, 34.114201423219896, 33.65297970469156, 33.38947254743721, 32.89797995748901, 32.65895472762013, 32.413028725189584, 32.149608572549866, 31.951295744008956, 31.68675873165685, 31.434510160159235, 31.08567600227998, 30.880741494707483, 30.549843176725634, 30.278037722375394, 30.116700900123373, 29.8823304524671, 29.51368456291673, 29.37089076099157, 29.10621008402717, 28.85311559588857, 28.686642624585534, 28.359956915602066, 28.055477546158166, 27.801930803398786]



# plt.plot(list21, bah5, color='#071ef0', linewidth=1.0, label='Benchmark')
# plt.plot(list21, b5, color='#e81809', linewidth=1.0, label='Proposed')
# plt.plot(list21, bah6, color='#071ef0', linewidth=1.0)
# plt.plot(list21, b6, color='#e81809', linewidth=1.0)
# plt.plot(list2, bah4, color='#071ef0', linewidth=1.0)
# plt.plot(list2, b4, color='#e81809', linewidth=1.0)
# plt.plot(list2, bah3, color='#071ef0', linewidth=1.0)
# plt.plot(list2, b3, color='#e81809', linewidth=1.0)
# plt.plot(list2, bah2, color='#071ef0', linewidth=1.0)
# plt.plot(list2, b2, color='#e81809', linewidth=1.0)
# plt.plot(list2, bah1, color='#071ef0', linewidth=1.0)
# plt.plot(list2, b1, color='#e81809', linewidth=1.0)



#plt.plot(list60, bah, color='#071ef0', linewidth=1.0, label='Benchmark')
plt.plot(list60, bil, color='#e81809', linewidth=1.0, label='Proposed')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()