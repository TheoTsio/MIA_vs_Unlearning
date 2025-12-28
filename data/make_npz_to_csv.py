import numpy as np
import pandas as pd

# Φόρτωση του αρχείου
data = np.load('purchase100.npz')
X = data['features']
y_one_hot = data['labels'] # Αυτό είναι το (197324, 100)

# ΜΕΤΑΤΡΟΠΗ: Από One-Hot σε 1D array με τα νούμερα των κλάσεων
# Το argmax βρίσκει τη θέση του "1" σε κάθε σειρά
y_numeric = np.argmax(y_one_hot, axis=1) 

# Δημιουργία του DataFrame
df = pd.DataFrame(X)

# Εισαγωγή των labels στην πρώτη στήλη
# Προσθέτουμε +1 αν ο κώδικας MIA που έχετε κάνει "y - 1" αργότερα
df.insert(0, 'label', y_numeric + 1) 

# Αποθήκευση
df.to_csv('dataset_purchase.csv', index=False)
print("Επιτυχής δημιουργία του dataset_purchase.csv!")
print(f"Νέο σχήμα labels: {y_numeric.shape}")