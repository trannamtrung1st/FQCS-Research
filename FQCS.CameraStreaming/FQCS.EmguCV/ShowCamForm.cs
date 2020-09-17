using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FQCS.EmguCV
{
    public partial class ShowCamForm : Form
    {
        private VideoCapture _vidCapture;
        public ShowCamForm()
        {
            InitializeComponent();
            this.FormClosed += ShowCamForm_FormClosed;
        }

        private void ShowCamForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (_vidCapture != null)
                _vidCapture.Dispose();
        }

        protected void ShowCam(string ipAddr)
        {
            if (_vidCapture != null) _vidCapture.Dispose();
            _vidCapture = string.IsNullOrWhiteSpace(ipAddr) ?
                new VideoCapture(0) : new VideoCapture(ipAddr);
            _vidCapture.ImageGrabbed += _vidCapture_ImageGrabbed;
            _vidCapture.Start();
        }

        private void _vidCapture_ImageGrabbed(object sender, EventArgs e)
        {
            using (Mat imageMat = new Mat())
            {
                _vidCapture.Read(imageMat);
                camBox.Image = imageMat.ToImage<Bgr, byte>().ToBitmap();
            }
        }

        private void btnShow_Click(object sender, EventArgs e)
        {
            string ipAddr = txtIpAddress.Text;
            ShowCam(ipAddr);
        }
    }
}
