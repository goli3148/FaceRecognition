from typing import Optional
import sqlalchemy as sa
import sqlalchemy.orm as so
from app import db

class People(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    label: so.Mapped[int] = so.mapped_column(sa.Integer(), index=True, nullable=False)
    image: so.Mapped[str] = so.mapped_column(sa.String(200), nullable=False)
    name: so.Mapped[str] = so.mapped_column(sa.String(50), nullable=True, default='UNKNOWN')

    def __repr__(self):
        return '<Individual {}>'.format(self.id, self.label, self.name, self.image)