import matplotlib.pyplot as plt
import utilities

ang = "E:/cells/CoNi90-OrthoCells_20240320_27061_scan3.ang"
ang_obj = utilities.read_ang(ang, None, 5.0)

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].imshow(ang_obj.iq)
ax[0, 0].set_title("IQ")
ax[0, 1].imshow(ang_obj.ci)
ax[0, 1].set_title("CI")
ax[1, 0].imshow(ang_obj.ids)
ax[1, 0].set_title("IDS")
ax[1, 1].imshow(ang_obj.kam)
ax[1, 1].set_title("KAM")
plt.tight_layout()
plt.show()
