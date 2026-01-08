// main/routes/report.js
const express = require('express');
const router = express.Router();
router.post('/generate', (req, res) => {
  res.json({ success: true, pdf_path: './reports/test.pdf' });
});
module.exports = router;